import jax 
from jax import vmap, lax
import jax.numpy as jnp

import numpy as np
from itertools import product
from functools import partial

###########
# project #
###########
@partial(jax.jit, static_argnums=(3,4,5,6))
def project(volume: 'NDArray[D,H,W]', thetas: 'NDArray[K]', phis: 'NDArray[K]', max_theta=60.0, voxel_size=(1,1,1), oversample=1, interp_method='quadratic') -> 'NDArray[K,H,W]':
    """ Differentiable stretched radon transform 
    
    Args
        volume: (depth, height, width) volume through which we project
        thetas: (degrees) tilt angles
        phis: (degrees) rotation in xy-plane of the tilt axis
        max_theta: (degrees) largest theta that will be given to project
            This is necessary as JAX requires knowing sizes of all tensor in order to JIT-compile
            Thetas are clipped to be within (-min_theta,max_theta)
        voxel_size: (depth,height,width) dimensions of voxels. 
        oversample: int, number of points-per xy section used to evaluate density along ray
        interp_method: Method method used to interpolate through volume
            'nearest', 'linear', or 'quadratic'
        
    Returns
        projection: (n_tilts, height, width) tensor of projections
        
    Notes
        projection is differentiable w.r.t. volume, thetas, and phis
    
    Example
        >>> n_tilts, depth, height, width = 45, 32, 500, 500
        >>> thetas = jnp.linspace(-45.,45.,n_tilts)
        >>> phis = jnp.zeros(n_tilts)
        >>> volume = jnp.ones((depth, height, width))
        >>> projection = project(x, thetas, phis)
        >>> print(projection.shape)
        (45,32,500,500)
    
    Implementation notes
        This is equivalent to convolving a dense 3D kernel with the input.
        For efficiency, we note that the kernel is sparse and break up the convolution into 
        a series of sparse convolutions at each depth in the volume, then offset these and 
        sum to generate a tilt
    """   
    # compute kernel size
    thetas = jnp.clip(thetas,-max_theta,max_theta)
    kernel_size = get_minimal_kernel_size(volume.shape[0], max_theta)
    
    # create sparse kernel 
    sparse_kernel, grid = create_sparse_kernel(thetas, phis, kernel_size, voxel_size, oversample, interp_method)
    corners = grid[:,:,:,0,0]
    
    # apply sparse kernel
    project_fn = vmap(vmap(sparse_conv, in_axes=(None,0,0,None)), in_axes=(None,0,0,None))
    projection = project_fn(volume,sparse_kernel,corners,kernel_size).mean(0) # (average is for oversampling)
    
    return projection

def sparse_conv(volume: 'NDArray[D,H,W]', kernel: 'NDArray[D,S,S]', offsets: 'NDArray[D,2]', dense_kernel_size: '(h,w)')-> 'NDArray[D,H,W]':
    """ Apply sparse convolutional with same padding
    
    Args
        volume: (depth, height, width) volume through which we project
        kernel: (depth, sparse_height, sparse_width) kernel we apply
        offsets: height, width offset of the kernel at every depth 
        dense_kernel_size (height, width) extent of the full dense kernel
        
    Returns
        conv: Result of applying sparse conv to volume. Equivalent to applying dense conv defined by (kernel, offsets)
    """
    padh = dense_kernel_size[0]-1
    padw = dense_kernel_size[1]-1
    padded = jnp.pad(volume, ((0,0), (padh//2,padh-padh//2), (padw//2, padw-padw//2)))
    extract_size = (volume.shape[1]+kernel.shape[1]-1, volume.shape[2]+kernel.shape[2]-1)
    shifted = vmap(lax.dynamic_slice,in_axes=(0,0,None))(padded,offsets,extract_size)
    p = lax.conv(shifted[None], kernel[None], window_strides=(1,1), padding='valid')[0,0]
    return p

#############################
# helper with kernel sizing #
#############################
def get_minimal_kernel_size(depth, max_theta):
    """ Compute minimal odd-intere kernel size gauranteed to contain every interpolation point """
    extent = depth * np.tan(np.pi / 180 * max_theta) + 3
    extent = 2 * np.floor(extent/2) + 1
    kernel_size = (depth, int(extent), int(extent))
    return kernel_size

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

def interpolation_weights(points: '...x2', kernel='quadratic') -> '...xSxSx2, ...xSxS':
    # get weights and locations of neighbors
    if kernel == 'nearest':
        edges = jnp.array(list(product((0,),(0,))),dtype='int16').reshape((1,1,2)) # 1x2
        reference = jnp.floor(points+0.5).astype('int16')# ...x1x1x2
        grid = reference[...,None,None,:] + edges # ...x1x1x2
        weights = jnp.ones((points.shape[:-1],1,1)) # ...x1x1
    if kernel == 'linear':
        edges = jnp.array(list(product((0,1),(0,1))),dtype='int16').reshape((2,2,2)) # 4x2
        reference = jnp.floor(points).astype('int16')
        grid = reference[...,None,None,:] + edges # ...x2x2x2
        weights = linear_kernel(grid[...,0]-points[...,None,None,0]) \
                * linear_kernel(grid[...,1]-points[...,None,None,1]) # HxWx2x2
    elif kernel == 'quadratic':    
        edges = jnp.array(list(product((-1,0,1),(-1,0,1))),dtype='int16').reshape((3,3,2)) # 9x2
        reference = jnp.floor(points+0.5).astype('int16') #...x2
        grid = reference[...,None,None,:] + edges # ...x3x3x2
        weights = quadratic_kernel(grid[...,0]-points[...,None,None,0]) \
                * quadratic_kernel(grid[...,1]-points[...,None,None,1]) # ...x3x3
    else:
        raise ValueError(f'Unrecognized kernel: {kernel}, must be "nearest", "linear", or "quadratic"')
    
    return grid, weights

##########
# kernel #
##########
def create_sparse_kernel(thetas: 'K', phis: 'K', kernel_size: '(D,H,W)', voxel_size=(1.0,1.0,1.0), oversample=1, interp_method='quadratic') -> 'OxKxDxHxW':
    """ Returns sparse kernel and offsets for theta and phis """
    return vmap(create_sparse_kernel_,in_axes=(0,0,None,None,None,None), out_axes=1)(thetas, phis, kernel_size, voxel_size, oversample, interp_method)

def create_sparse_kernel_(theta: float, phi: float, kernel_size: '(D,H,W)', voxel_size: '(d,h,w)', oversample: int, interp_method: str) -> 'OxDxSxS, OxDx2':
    """ Returns sparse kernel Q and offsets for a single theta and phi """
    points = integration_points(theta, phi, kernel_size, voxel_size, oversample)
    grid, weights = interpolation_weights(points[...,1:], interp_method)
    corners = grid.min((2,3)) # idenify corners of grid
    weights = weights / jnp.cos(theta*np.pi/180.) # normalize by segment len
    
    return weights, grid#corners 

def sparse2dense(weights: 'OxKxDxSxS', grid: 'OxKxDxSxSx2', kernel_size: 'H,W') -> 'KxDxHxW':
    """ Convert sparse kernel to dense kernel """
    O,K,D = weights.shape[:3]
    kernel = jnp.zeros((O,K,D)+kernel_size)
    kernel = kernel.at[jnp.arange(O)[:,None,None,None,None], jnp.arange(K)[None,:,None,None,None], jnp.arange(D)[None,None,:,None,None], grid[...,0], grid[...,1]].set(weights)
    kernel = kernel.mean(0)
    return kernel
