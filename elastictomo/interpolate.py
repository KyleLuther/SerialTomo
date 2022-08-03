""" Operations for working with continuous images 
    interpolate: maps scalar field defined on grid to continuous scalar field
    interpolate_vecs: maps vector field defined on grid to continuous vector field
    pixelate: maps continuous scalar field to grid by integrating inside pixels
"""
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from flax.linen import avg_pool

def interpolate(grid):
    """ Returns a function which linearly interpolates values on grid 
    
    Args : 
        grid : n-dim array of values to interpolate
    Returns :
        interpolated_grid : R^n -> R, a function which interpolates grid at continuous locations 
            Accepts arrays of shape (*,n) and returns array of shape (*)
            Assumes grid is defined at center of each pixel (y+0.5,x+0.5)
    Example :
        >>> grid = jnp.ones((3,4,5))
        >>> volume = interpolate(grid)
        >>> print(grid[0,0,0])
        Output: 1.0
        >>> print(volume(jnp.array((0.5,0.5,0.5))))
        Output: 1.0
    """
    def interpolated(r):
        # mask out regions outside image
        mask = jnp.ones(r.shape[:-1])
        for i in range(len(grid.shape)):
            mask = mask * (r[...,i] >= 0.0) * (r[...,i] <= grid.shape[i])

        # use center pixels as the value
        padded = jnp.pad(grid,1,mode='edge') 
        values = map_coordinates(padded, jnp.moveaxis(r,-1,0)+0.5, order=1)

        return mask*values
    
    return interpolated

def interpolate_vecs(grid):
    """ Returns a function which linearly interpolates n-dim vectors on grid 
    
    Args : 
        grid: (n+1)-dim array, shape of final dimension is vector dimension, d
    Returns :
        interpolated_grid: R^n -> R^d, a function which interpolates grid at continuous locations 
    
    Example :
        >>> grid = np.random.randn((3,4,5,2))
        >>> print(grid[0,0,0])
        Output: 0.83929193, 0.1234321
        >>> volume = interpolate_nd(grid)
        >>> print(volume(jnp.array((0.5,0.5,0.5))))
        Output: 0.83929193, 0.1234321
    Notes : 
        interpolated_grid accepts arrays of shape (*,n) and returns array of shape (*,d)
        assumes grid is defined at center of each pixel (y+0.5,x+0.5)
    """
    grids = [interpolate(grid[...,i]) for i in range(grid.shape[-1])]
    def interpolated_vecs(r):
        return jnp.stack([g(r) for g in grids],axis=-1)
    return interpolated_vecs

def pixelate(img, grid_size, oversample):
    """ Integrates img into unit boxes to generate pixel image
    Args : 
        img : continuous image operator R^n -> R
        grid_size : (ny,nx) number of output pixels
        oversample: (fy,fx) fy*fx is number of integration points for each pixel cell
    Returns :
        pixelated_img : (ny,nx) array of integrated pixel values
    Notes :
        Returns crop from (0,0) to grid_size
    """
    center = tuple(s/2 for s in grid_size)
    gs = []
    for i in range(len(grid_size)):
        g = jnp.linspace(0.5/oversample[i], 
                         grid_size[i] - 0.5/oversample[i], 
                         grid_size[i]*oversample[i])
        gs.append(g)
    gs = jnp.meshgrid(*gs,indexing='ij')
    u = jnp.stack(gs,axis=-1)
    
    grid = avg_pool(img(u)[None,...,None],
                          window_shape=oversample,
                          strides=oversample)[0,...,0]
    return grid
