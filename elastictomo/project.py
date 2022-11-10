import jax 
from jax import vmap
from jax import lax

import jax.numpy as jnp
from jax.nn import one_hot
from jax.scipy.ndimage import map_coordinates

import numpy as np
from itertools import product
from functools import partial

######################
# integration points #
######################
def integration_points(theta: float, phi: float, kernel_size=(16,16,16), voxel_size=(1.0,1.0,1.0), oversample=1) -> jnp.ndarray:
    # create integration points in physical coordinates
    zs = jnp.arange(-kernel_size[0]/2,kernel_size[0]/2)
    zs = zs + ((1+2.0*np.arange(oversample))/(2*oversample))[:,None]
    zs = voxel_size[0] * zs # OxD

    ys = jnp.sin(phi*np.pi/180.0) * jnp.tan(theta*np.pi/180.0) * zs
    xs = jnp.cos(phi*np.pi/180.0) * jnp.tan(theta*np.pi/180.0) * zs
    rs = jnp.stack([zs,ys,xs], axis=-1) # OxDx3
    
    # map to kernel coordinates
    rs = rs / jnp.array(voxel_size)
    rs = rs + jnp.array(kernel_size) / 2.0
    
    return rs

###############
# interpolate #
###############
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

def interpolation_weights(points: 'OxDx2', kernel='quadratic') -> 'OxDxQx2, OxDxQ':
    # get weights and locations of neighbors
    if kernel == 'nearest':
        edges = jnp.array(list(product((0,),(0,))),dtype='int16') # 1x2
        reference = jnp.floor(points+0.5).astype('int16')# ...x1x2
        weights = jnp.ones((points.shape[:-1],1)) # ...x1
    if kernel == 'linear':
        edges = jnp.array(list(product((0,1),(0,1))),dtype='int16') # 4x2
        reference = jnp.floor(points).astype('int16')
        grid = reference[...,None,:] + edges # ...x4x2
        weights = linear_kernel(grid[...,0]-points[...,None,0]) \
                * linear_kernel(grid[...,1]-points[...,None,1]) # HxWx4
    elif kernel == 'quadratic':    
        edges = jnp.array(list(product((-1,0,1),(-1,0,1))),dtype='int16') # 9x2
        reference = jnp.floor(points+0.5).astype('int16') #...x2
        grid = reference[...,None,:] + edges # ...x9x2
        weights = quadratic_kernel(grid[...,0]-points[...,None,0]) \
                * quadratic_kernel(grid[...,1]-points[...,None,1]) # ...x9
    else:
        raise ValueError(f'Unrecognized kernel: {kernel}, must be "linear" or "quadratic"')
    
    return grid, weights

##########
# kernel #
##########
def weights2kernel(grid: 'OxDxQx2', weights: 'OxDxQ', kernel_size: 'D,H,W') -> 'DxHxW':
    oversample = grid.shape[0]
    kernel = jnp.zeros((oversample,)+kernel_size)
    kernel = kernel.at[jnp.arange(oversample)[:,None,None], jnp.arange(kernel_size[0])[None,:,None], grid[...,0], grid[...,1]].set(weights)
    kernel = kernel.sum(0)
    return kernel

def normalize_weights(grid: 'OxDxQx2', weights: 'OxDxQ', theta: float, img_size: 'H,W'):
    """ Normalize weights to A) handle out of bounds B) correct for segment lengths"""
    # handle out of bounds
    in_bounds = (grid[...,0] >= 0) * (grid[...,0] <= img_size[0]-1) *\
                (grid[...,1] >= 0) * (grid[...,1] <= img_size[1]-1)
    norm = (in_bounds * weights).sum(-1,keepdims=True) * grid.shape[1]
    
    # correct for segment length
    norm = jnp.cos(theta*np.pi/180.) * norm
    
    return weights / norm
    
def create_kernel_(theta: float, phi: float, kernel_size: 'D,H,W', voxel_size: 'd,h,w', oversample: int, interp_method: str) -> 'DxHxW':
    points = integration_points(theta, phi, kernel_size, voxel_size, oversample)
    grid, weights = interpolation_weights(points[...,1:], interp_method) # interpolate in xy only
    weights = normalize_weights(grid, weights, theta, kernel_size[1:])
    kernel = weights2kernel(grid, weights, kernel_size)

    return kernel

@partial(jax.jit, static_argnums=(2,3,4,5))
def create_kernel(thetas: 'K', phis: 'K', kernel_size: 'D,H,W', voxel_size=(1.0,1.0,1.0), oversample=1, interp_method='quadratic') -> 'KxDxHxW':
    return vmap(create_kernel_,in_axes=(0,0,None,None,None,None))(thetas, phis, kernel_size, voxel_size, oversample, interp_method)

###########
# project #
###########
@partial(jax.jit, static_argnums=(3,4,5,6))
def project(x: 'DxHxW', thetas: 'K', phis: 'K', kernel_size=(16,16,16), voxel_size=(1,1,1), oversample=1, interp_method='quadratic') -> 'KxHxW':
    kernel = create_kernel(thetas, phis, kernel_size, voxel_size, oversample, interp_method)
    y = jax.lax.conv(x[None], kernel, window_strides=(1,1), padding='valid')[0]
    
    return y