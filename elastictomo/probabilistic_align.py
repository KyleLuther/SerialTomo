import jax 
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, jacfwd, jacrev
from jax.scipy.ndimage import map_coordinates
import numpy as np

from elastictomo.optimize import minimize
from tqdm import tqdm, trange


################
# registration #
################
def register_pair(ref: 'HxW float32', mov: 'HxW float32', grid_size=(2,2), init_position=(0,0), **kwargs):
    # checks
    assert((ref.dtype == 'float32') and (mov.dtype == 'float32'))
    assert((ref.ndim == 2) and (mov.ndim == 2))

    # gradient functions
    vng_fun = jit(value_and_grad(energy))
    fun = lambda d: vng_fun(d, ref, mov)
    
    # initialize displacements
    offset = jnp.array((mov.shape[0] - ref.shape[0], mov.shape[1] - ref.shape[1]), dtype='float32') / 2.0
    displacements = jnp.zeros((grid_size[0], grid_size[1],2)) + offset + jnp.array(init_position)
    
    # optimize displacements
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

###############################
# probabilistic interpolation #
###############################
def upsample_displacements(displacements: 'HxWx2 array', N: 'int') -> 'HNxWNx2 array':
    # create grid locations
    H, W, C = displacements.shape
    r = jnp.stack(jnp.meshgrid(jnp.linspace(0,H-1,N*(H-1)), jnp.linspace(0,W-1,N*(W-1)), jnp.arange(C), indexing='ij'), axis=-1)

    # sample grid locations
    displacements_interp = map_coordinates(displacements, r.transpose(3,0,1,2), order=1)
    
    return displacements_interp

def apply_displacements(mov: 'hxw array', displacements: 'HxWx2 array') -> 'HxWx4, HxWx4 array':
    # create grid locations
    H, W = displacements.shape[:2]
    r = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'), axis=-1).astype(displacements.dtype)
    r = r + displacements
    
    # create neighbor locations
    neighbors = jnp.zeros((H,W,4,2),dtype='int')
    neighbors = neighbors.at[:,:,0,0].set(jnp.floor(r[:,:,0])) # (0,0)
    neighbors = neighbors.at[:,:,0,1].set(jnp.floor(r[:,:,1])) # (0,0)
    neighbors = neighbors.at[:,:,1,0].set(jnp.floor(r[:,:,0])) # (0,1)
    neighbors = neighbors.at[:,:,1,1].set(jnp.floor(r[:,:,1])+1) # (0,1)
    neighbors = neighbors.at[:,:,2,0].set(jnp.floor(r[:,:,0])+1) # (1,0)
    neighbors = neighbors.at[:,:,2,1].set(jnp.floor(r[:,:,1])) # (1,0)
    neighbors = neighbors.at[:,:,3,0].set(jnp.floor(r[:,:,0])+1) # (1,1)
    neighbors = neighbors.at[:,:,3,1].set(jnp.floor(r[:,:,1])+1) # (1,1)
                           
    # get weights
    weights = jnp.zeros((H,W,4))
    weights = weights.at[:,:,0].set((r[:,:,0]-neighbors[:,:,0,0]-1) * (r[:,:,1] - neighbors[:,:,0,1]-1))
    weights = weights.at[:,:,1].set((r[:,:,0]-neighbors[:,:,1,0]-1) * (neighbors[:,:,1,1]-r[:,:,1]-1))
    weights = weights.at[:,:,2].set((neighbors[:,:,2,0]-r[:,:,0]-1) * (r[:,:,1] - neighbors[:,:,2,1]-1))
    weights = weights.at[:,:,3].set((neighbors[:,:,3,0]-r[:,:,0]-1) * (neighbors[:,:,3,1]-r[:,:,1]-1))
    
    # get values
    mov_interp = mov[neighbors[:,:,:,0], neighbors[:,:,:,1]] # H x W x 4
    
    # make sure to hand out of bounds values correctly
    in_bounds = (neighbors[:,:,:,0] >= 0).astype('float') *\
                (neighbors[:,:,:,0] <= mov.shape[0]-1).astype('float') *\
                (neighbors[:,:,:,1] >= 0).astype('float') *\
                (neighbors[:,:,:,1] <= mov.shape[1]-1).astype('float')
    
    mov_interp = mov_interp * in_bounds
    
    weights = weights * in_bounds

    return mov_interp, weights

########################
# probabilistic energy #
########################
def energy(displacements, ref, mov):
    # geometry
    H, W = ref.shape
    N = H // (displacements.shape[0]-1) # upsample factor
    
    # upsample displacement field and interpolate data
    upsampled = upsample_displacements(displacements, N)
    mov_interp, weights = apply_displacements(mov, upsampled)
    
    # compute energy
    # e = (weights*(mov_interp - ref[:,:,None])**2).sum()
    e = (weights*(mov_interp - ref[:,:,None])**2).sum() / weights.mean()
    
    return e