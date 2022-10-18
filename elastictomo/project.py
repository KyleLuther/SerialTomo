import jax 
from jax import vmap
from jax import lax

import jax.numpy as jnp
from jax.nn import one_hot
from jax.scipy.ndimage import map_coordinates

import numpy as np
from itertools import product

def rot_mat(theta):
    """ Creates Euler rotation matrix
    Args :
        theta : (tz,ty,tz) in degrees
    Returns:
        R : (3,3) matrix equal to R_z @ R_x @ R_y
    """
    theta_z, theta_y, theta_x = tuple(-o*np.pi/180 for o in theta)
    Rx_ = jnp.array([[1,0,0],[0,jnp.cos(theta_x),-jnp.sin(theta_x)],[0,jnp.sin(theta_x),jnp.cos(theta_x)]])
    Rz_ = jnp.array([[jnp.cos(theta_z),-jnp.sin(theta_z),0],[jnp.sin(theta_z),jnp.cos(theta_z),0],[0,0,1]])
    Ry_ = jnp.array([[jnp.cos(theta_y),0,jnp.sin(theta_y)],[0,1,0],[-jnp.sin(theta_y),0,jnp.cos(theta_y)]])
    R_ = Rz_ @ Rx_ @ Ry_
    R = jnp.flip(R_,axis=(0,1))
    return R

def integration_weights(r, kernel_size):
    """ Creates interpolation weights for r in volume """
    # get weights to each grid point
    r = r - 0.5
    corners = jnp.floor(r)[...,None,:] + jnp.array(tuple(product((0,1),repeat=3))) # ... x 8 x 3
    weights = jnp.prod(1 - jnp.abs(r[...,None,:] - corners), axis=-1) # ... x 8
    
    # mask out of bounds corners
    mask = jnp.ones(weights.shape)
    for i in range(3):
        mask = mask * (corners[...,i] >= 0) * (corners[...,i] < kernel_size[i])
    weights = (mask*weights)
    
    # rescale at the borders
    weights = weights / weights.sum(-1, keepdims=True).clip(1/8)
    
    # map grid points to 1 hot vectors
    factor_ = jnp.array([kernel_size[1]*kernel_size[2], kernel_size[2], 1]) # 3
    locs = (corners*factor_).sum(axis=-1).astype(int) # ... x 8
    ones = one_hot(locs, np.prod(kernel_size)) # ... x 8 x np.prod(kernel_size)
    
    # multiply by weights and sum
    grid_weights = (ones * weights[...,None]).reshape((-1,)+kernel_size) # ... x kernel_size
    integ_weights = grid_weights.sum(0) # kernel_size

    return integ_weights

def integration_weights_nearest(r, kernel_size):
    """ Creates interpolation weights for r in volume """
    # get weights to each grid point
    r = r - 0.5
    corners = jnp.floor(r)[...,None,:] + jnp.array(tuple(product((0,1),repeat=3))) # ... x 8 x 3
    weights = jnp.prod(1 - jnp.abs(r[...,None,:] - corners), axis=-1) # ... x 8
    
    # mask out of bounds corners
    mask = jnp.ones(weights.shape)
    for i in range(3):
        mask = mask * (corners[...,i] >= 0) * (corners[...,i] < kernel_size[i])
    weights = (mask*weights)
    
    # rescale at the borders
    weights = weights / weights.sum(-1, keepdims=True).clip(1/8)
    
    # map grid points to 1 hot vectors
    factor_ = jnp.array([kernel_size[1]*kernel_size[2], kernel_size[2], 1]) # 3
    locs = (corners*factor_).sum(axis=-1).astype(int) # ... x 8
    ones = one_hot(locs, np.prod(kernel_size)) # ... x 8 x np.prod(kernel_size)
    
    # multiply by weights and sum
    grid_weights = (ones * weights[...,None]).reshape((-1,)+kernel_size) # ... x kernel_size
    integ_weights = grid_weights.sum(0) # kernel_size

    return integ_weights

def projection_kernel_(theta=(0.0,0.0,0.0), kernel_size=(16,16,16), voxel_size=(1,1,1), offset=(0.0,0.0), oversample=1, normalize=True):
    """ Creates projection kernel """
    # checks
    assert(len(theta) == len(kernel_size) == len(voxel_size) == 3)
    assert(len(offset) == 2)
    
    # unpack
    D, H, W = (k*s for k,s in zip(kernel_size, voxel_size)) # physical dimensions of kernel
    
    # find endpoints of line segment in physical coordinates
    R = rot_mat(theta)
    z = D/2
    y = R[1,0] / R[0,0] * D/2
    x = R[2,0] / R[0,0] * D/2
    
    # seg len
    n_points = int(kernel_size[0]*oversample)
    line_len = 2*jnp.sqrt(x**2 + y**2 + z**2)
    seg_len = line_len / n_points
        
    # create integration points in physical coordinates
    zs = jnp.linspace(-z, +z, n_points, endpoint=False)
    zs = zs + (z-zs[-1]) / 2.0
    
    ys = jnp.linspace(-y, +y, n_points, endpoint=False)
    ys = ys + (y-ys[-1]) / 2.0
    
    xs = jnp.linspace(-x, +x, n_points, endpoint=False)
    xs = xs + (x-xs[-1]) / 2.0
    
    rs = jnp.stack([zs,ys,xs], axis=-1)
    
    # map physical coordinates to grid coordinates
    rs = rs / jnp.array(voxel_size)
    rs = rs + jnp.array(kernel_size) / 2.0 + jnp.array([0.0,offset[0],offset[1]])
    
    # map grid coordinates to interpolation weights
    kernel = integration_weights(rs, kernel_size)

    # normalize
    if normalize:
        kernel = 1 / kernel.sum() * kernel
    else: 
        kernel = seg_len / D * kernel
    
    return kernel

def projection_kernel(thetas, kernel_size=(16,16,16), voxel_size=(1,1,1), offset=(0.0,0.0), oversample=1, normalize=True):
    return vmap(projection_kernel_, in_axes=(0,None,None,None,None,None))(thetas, kernel_size, voxel_size, offset, oversample, normalize)

def project(x, P):
    y = lax.conv(x[None], P, window_strides=(1,1), padding='valid')[0]
    return y

# def minimal_kernel_size(thetas, D):
#     """ Returns smallest kernel size that contains full kernel """
#     H = W = int(np.ceil(D / np.cos(thetas.max())))
#     p = projection_kernel(thetas, kernel_size=(D,H,W), voxel_size=(1,1,1), offset=(0.0,0.0), oversample=1)
#     size_y = (p.sum(axis=(0,1,3)) > 0).sum()
#     size_x = (p.sum(axis=(0,1,2)) > 0).sum()
    
#     return size_y, size_x
    

# def project(vol, thetas, kernel_size, voxel_size, offset, oversample=1, normalize=True, interp_method='trilinear'):
#     # create projection kernel
#     P = projection_kernel(thetas, kernel_size, voxel_size)
    
#     # create projection
#     c = lax.conv(vol[None],P,window_strides=(1,1),padding='valid')[0]
 
#     return c