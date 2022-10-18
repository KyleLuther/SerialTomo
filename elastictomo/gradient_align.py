import jax 
from jax import lax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, jacfwd, jacrev
from jax.scipy.ndimage import map_coordinates
import numpy as np
from itertools import product
from tqdm import tqdm, trange

from elastictomo.optimize import minimize

################
# registration #
################
def register_pair(ref: 'HxW float32', mov: 'HxW float32', grid_size=(2,2), lam=1.0, init_position=(0,0), method='interpolate_energy', kernel='quadratic', **kwargs):
    # checks
    assert((ref.ndim == 2) and (mov.ndim == 2))
    ref, mov = jnp.array(ref), jnp.array(mov)
    
    # initialize displacements to be in center + user provided displacement
    init_offset = jnp.array((mov.shape[0] - ref.shape[0], mov.shape[1] - ref.shape[1]), dtype='float32') / 2.0
    displacements = jnp.zeros((grid_size[0],grid_size[1],2)) + init_offset + jnp.array(init_position)
    
    # optimize displacements
    energy_ = lambda displacements: energy(displacements, ref, mov, lam, method, kernel)
    fun = jit(value_and_grad(energy_))
    displacements, info = minimize(fun, displacements, **kwargs)
    
    return displacements, info
    
##################
# transformation #
##################
def transform_stack(mov: 'Kxhxw array', displacements: 'KxHxWx2 array', N) -> 'KxNxM array':
    out = np.zeros((mov.shape[0], (displacements.shape[1]-1)*N, (displacements.shape[2]-1)*N))
    for i in trange(mov.shape[0]):
        out[i] = np.array(transform_img(mov[i], displacements[i], N))
    return out
        
def transform_img(mov: 'hxw array', displacements: 'HxWx2 array', N) -> 'NxM array':
    upsampled = upsample_displacements(displacements, N)
    corners, weights = apply_displacements(mov, upsampled)
    return (weights*corners).sum(-1)

#########################
# interpolation kernels #
#########################
def linear_kernel(s: jnp.array) -> jnp.array:
    """ linear interpolation kernel
    Ref: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6893&rep=rep1&type=pdf
    """
    s = jnp.abs(s)
    return jnp.where(s <= 1, 1-s, jnp.zeros(s.shape,dtype=s.dtype))

def quadratic_kernel(s: jnp.array) -> jnp.array:
    """ quadratic interpolation kernel
    Ref: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6893&rep=rep1&type=pdf
    """
    def f1(s):
        return -s**2 + 3/4
    def f2(s):
        return 1/2 * s**2 - 3/2 * s + 9/8
    
    s = jnp.abs(s)
    out = jnp.where(s <= 1/2, f1(s), f2(s))
    out = jnp.where(s > 3/2, jnp.zeros(s.shape,dtype=s.dtype), out)
    return out

#########################
# interpolation weights #
#########################
def interpolate(img: 'NxM array', displacements: 'HxWx2 array', kernel='quadratic') -> 'HxWxQ, HxWxQ, HxWxQ array':
    """ Returns grid points and interpolation points for 2D image interpolation with displacement fields
    Args:
        img: NxM array
        displacements: HxW array
        kernel: (str) "linear" or "quadratic"
    Returns:
        interp: HxWxQ array. Image interpolated at grid points
        weights: HxWxQ float32. Weights of interpolated grid points
        in_bounds: HxWxQ bool. Specifies if grid points were in bounds
        
        Q is the number of interpolation points. 
        Q=4 for linear kernel, Q=9 for quadratic kernel
    """
    # get weights and locations of neighbors
    if kernel == 'linear':
        edges = jnp.array(list(product((0,1),(0,1))),dtype='int16') # 4x2
        reference = jnp.floor(displacements).astype('int16')
        weights = linear_kernel(reference[:,:,None,0]-displacements[:,:,None,0]+edges[...,0]) \
                * linear_kernel(reference[:,:,None,1]-displacements[:,:,None,1]+edges[...,1]) # HxWx4
    elif kernel == 'quadratic':    
        edges = jnp.array(list(product((-1,0,1),(-1,0,1))),dtype='int16') # 9x2
        reference = jnp.floor(displacements+0.5).astype('int16')
        weights = quadratic_kernel(reference[:,:,None,0]-displacements[:,:,None,0]+edges[...,0]) \
                * quadratic_kernel(reference[:,:,None,1]-displacements[:,:,None,1]+edges[...,1]) # HxWx9
    else:
        raise ValueError(f'Unrecognized kernel: {kernel}, must be "linear" or "quadratic"')
        
    # create grid locations
    H, W = displacements.shape[:2]
    r = jnp.stack(jnp.meshgrid(jnp.arange(H,dtype='int16'), jnp.arange(W,dtype='int16'), indexing='ij'), axis=-1)
    grid = r[:,:,None,:] + reference[:,:,None,:] + edges[None,None,:,:] # HxWxQx2

    # access neighbors
    interp = img[grid[...,0], grid[...,1]]
    
    # make sure to hand out of bounds values correctly
    in_bounds = (grid[...,0] >= 0) * (grid[...,0] <= img.shape[0]-1) *\
                (grid[...,1] >= 0) * (grid[...,1] <= img.shape[1]-1)
    
    return interp, weights, in_bounds

##########
# energy #
##########
def energy(displacements, ref, mov, lam, method, kernel):
    # upsample displacement field
    H, W = ref.shape
    displacements = jax.image.resize(displacements, (H,W,2), method='cubic')
    
    return photometric_energy(displacements,ref,mov,method,kernel) + lam * elastic_energy(displacements)
    
def photometric_energy(displacements: 'HxWx2 array', ref: '2D array', mov: '2D array', method='interpolate_energy', kernel='quadratic') -> float:
    # checks
    assert((displacements.shape[0] == ref.shape[0]) and (displacements.shape[1] == ref.shape[1]))
    
    # interpolation weights
    mov_interp, weights, in_bounds = interpolate(mov, displacements, kernel=kernel)
    
    # compute energy
    if method == 'interpolate_energy':
        e = (in_bounds*weights*(mov_interp - ref[...,None])**2).sum() / in_bounds.mean()
    elif method == 'interpolate_image':
        e = (in_bounds.all(axis=-1)*((weights*mov_interp).sum(-1) - ref)**2).sum() / in_bounds.all(axis=-1).mean()
    else:
        raise ValueError(f'Unrecognized method {method}')
    
    return e

def elastic_energy(displacements: 'array(H,W,2)') -> float:
    # distances along vertical and horizontal edges
    dys = jnp.linalg.norm(jnp.array([1.,0.]) + displacements[1:] - displacements[:-1], axis=-1)
    dxs = jnp.linalg.norm(jnp.array([0.,1.]) + displacements[:,1:] - displacements[:,:-1], axis=-1)
    
    # distances along diagonal edges
    das = jnp.linalg.norm(jnp.array([1.,1.]) + displacements[1:,1:] - displacements[:-1,:-1], axis=-1)
    dbs = jnp.linalg.norm(jnp.array([-1.,1.]) + displacements[:-1,1:] - displacements[1:,:-1], axis=-1)
    
    # compute energy
    e = ((dys-dys.mean())**2).sum() + ((dxs-dxs.mean())**2).sum() + ((das-das.mean())**2).sum() + ((dbs-dbs.mean())**2).sum()
    
    return e

###############
# In progress #
###############
def randomized_interpolate(key: 'PRNGKey', img: 'hxw array', displacements: 'HxWx2 array', kernel='quadratic', R=1) -> 'HxWxR, HxWxR, HxWxR array':
    """ Returns grid points and interpolation points for 2D image interpolation with displacement fields
    Args:
        key: prng key
        img: NxM array
        displacements: HxW array
        kernel: (str) "linear" or "quadratic"
        R: number of edges per grid point to sample
    Returns:
        interp: HxWxR array. Image interpolated at grid points
        weights: HxWxR float32. Weights of interpolated grid points
        in_bounds: HxWxR bool. Specifies if grid points were in bounds
    """
    # get weights and locations of neighbors
    H, W = displacements.shape[:2]
    if kernel == 'linear':
        reference = jnp.floor(displacements).astype('int16')
        edges = jax.random.choice(key,jnp.array([0,1],dtype='int8'),shape=(H,W,R,2))
        weights = linear_kernel(reference[:,:,None,0]-displacements[:,:,None,0]+edges[...,0]) \
                * linear_kernel(reference[:,:,None,1]-displacements[:,:,None,1]+edges[...,1]) # HxWxR
    elif kernel == 'quadratic':    
        reference = jnp.floor(displacements+0.5).astype('int16')
        edges = jax.random.choice(key,jnp.array([-1,0,1],dtype='int8'),shape=(H,W,R,2))
        weights = quadratic_kernel(reference[:,:,None,0]-displacements[:,:,None,0]+edges[...,0]) \
                * quadratic_kernel(reference[:,:,None,1]-displacements[:,:,None,1]+edges[...,1]) # HxWxR
    else:
        raise ValueError(f'Unrecognized kernel: {kernel}, must be "linear" or "quadratic"')
        
    # create grid locations
    r = jnp.stack(jnp.meshgrid(jnp.arange(H,dtype='int16'), jnp.arange(W,dtype='int16'), indexing='ij'), axis=-1)
    grid = r[:,:,None,:] + reference[:,:,None,:] + edges

    # access neighbors
    interp = img[grid[...,0], grid[...,1]]
    
    # make sure to hand out of bounds values correctly
    in_bounds = (grid[...,0] >= 0) * (grid[...,0] <= img.shape[0]-1) *\
                (grid[...,1] >= 0) * (grid[...,1] <= img.shape[1]-1)

    return interp, weights, in_bounds