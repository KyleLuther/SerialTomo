""" Code to simulate forward projection """

import jax 
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.scipy.ndimage import map_coordinates
from flax.linen import avg_pool

import numpy as np
from scipy.spatial.transform import Rotation

# helper functions
def rot_mat(theta):
    """ Creates Euler rotation matrix for xyz angles """
    theta_z, theta_y, theta_x = tuple(-o*np.pi/180 for o in theta)
    Rx_ = jnp.array([[1,0,0],[0,jnp.cos(theta_x),-jnp.sin(theta_x)],[0,jnp.sin(theta_x),jnp.cos(theta_x)]])
    Rz_ = jnp.array([[jnp.cos(theta_z),-jnp.sin(theta_z),0],[jnp.sin(theta_z),jnp.cos(theta_z),0],[0,0,1]])
    Ry_ = jnp.array([[jnp.cos(theta_y),0,jnp.sin(theta_y)],[0,1,0],[-jnp.sin(theta_y),0,jnp.cos(theta_y)]])
    R_ = Ry_ @ Rz_ @ Rx_
    R = jnp.flip(R_,axis=(0,1))
    return R

# helper functionals
def interpolate_grid(img_grid):
    """ Returns a function which linearly interpolates image_grid 
    Args : 
        img_grid: n-dim array
    Returns :
        img: a fn which interpolates img_grid at continuous locations
    """
    def interpolated_grid(r):
        """ Interpolate image at value r. 
        Args : 
            r : (*,n) array where n is n-dimension of img_grid
        Returns :
            values : array of size r.shape[:-1] of interpolated values
        Notes :
            Returns 0 if r is outside border. img_grid gives values of pixels
            at the center of a grid, so r=0.5 maps to img[0]
        """
        # mask out regions outside image
        mask = jnp.ones_like(r[...,0])
        for i in range(len(img_grid.shape)):
            mask = mask * (r[...,i] >= 0.0) * (r[...,i] < img_grid.shape[i])

        # use center pixels as the value
        padded = jnp.pad(img_grid,1,mode='edge') 
        values = map_coordinates(padded, jnp.moveaxis(r,-1,0)+0.5, order=1)

        return mask*values
    
    return interpolated_grid

def scale_img(img, scale):
    """ Rescale img 
    Args : 
        img : continuous image
        scale : (scale_z, scale_y, scale_x) 
    Returns :
        scaled_img : continuous image
    """
    scale = jnp.array(scale)
    def scaled_img(r):
        return img(r/scale)
    return scaled_img

def shift_img(img, shift):
    """ Shifts img 
    Args : 
        img : continuous image
        shift : (shift_z, shift_y, shift_x) 
    Returns :
        shifted_img : continuous image
    """
    shift = jnp.array(shift)
    def shifted_img(r):
        return img(r-shift)
    return shifted_img

def rotate_img(img, center, theta):
    """ Rotates img 
    Args : 
        img : continuous image operator
        center : (z,y,x) point around which image rotates
        theta : (theta_z, theta_y, theta_x) in degrees
    Returns :
        rotated_img : continuous image operator
    """
    center = jnp.array(center)
    def rotated_img(r):
        R = rot_mat(theta)
        return img((r-center) @ R.transpose() + center)
    return rotated_img

def warp_img2d(img, dy, dx):
    """ Warps 2d img
    Args : 
        img : continuous image R^2 -> R
        dy : continuous displacement operator R^2 -> R
        dx : continuous displacement operator R^2 -> R
    Returns :
        warped_img : continuous image operator
    """
    def warped_img(r):
        return img(r - jnp.stack([dy(r),dx(r)],axis=-1))
    return warped_img

def map_img(img, f):
    """ Apply f to the output of img """
    def mapped_img(r):
        return f(img(r))
    return mapped_img

def integrate_z(img, z_min, z_max, n_points):
    """ Integrates 3D image along z axis
    Args : 
        img : continuous image operator R^3 -> R
        z_min : continuous displacement operator R^2 -> R
        z_max : continuous displacement operator R^2 -> R
        n_points : number of points to use to integrate
    Returns :
        integral : 2d continuous image
    """
    def integrated_z(u):
        # evaluate bounds at the point u
        z_min_, z_max_ = z_min(u), z_max(u)
        
        # create points 
        seg_lens = jnp.abs(z_max_-z_min_) / n_points
        z = jnp.linspace(z_min_,z_max_,n_points,endpoint=False) + seg_lens[None]/2.0
        points = jnp.concatenate([z[...,None],jnp.repeat(u[None],n_points,0)],axis=-1)
        
        # evaluate 
        integral = (img(points) * seg_lens).sum(0)
        
        return integral
        
    return integrated_z

def pixelate_img(img, center, grid_size, oversample):
    """ Integrates img in yx to generate pixel image
    Args : 
        img : continuous image operator R^2 -> R
        center : (y,x) location of center
        grid_size : (ny, nx) number of output pixels
        oversample: (ny, nx), ny*nx is number of integration points for each pixel cell
    Returns :
        pixelated_img : 2d grid of pixel values
    """
    y_min, y_max = center[0] - grid_size[0]/2, center[0] + grid_size[0]/2
    x_min, x_max = center[1] - grid_size[1]/2, center[1] + grid_size[1]/2
    y = jnp.linspace(y_min+0.5/oversample[0], y_max-0.5/oversample[1], grid_size[0]*oversample[0])
    x = jnp.linspace(x_min+0.5/oversample[1], x_max-0.5/oversample[1], grid_size[1]*oversample[0])
    y,x = jnp.meshgrid(y,x,indexing='ij')
    u = jnp.stack([y,x],axis=-1)
    
    grid = avg_pool(img(u)[None,...,None],
                          window_shape=(oversample[0],oversample[1]),
                          strides=(oversample[0],oversample[1]))[0,...,0]
    return grid
    

# simulate propagation through a volume with lens warping
def project(vol_grid: "(D,H,W) array", displacement_grid: "(H,W,2) array", 
            orientation=(0.0,0.0,0.0), shift=(0.0,0.0),
            voxel_size=(1.0,1.0,1.0), ccd_size=(128,128), ray_oversample=(1,1,1)):
    """ Simulate electron transmission and image acquisition through 3d volume
    
    Refer to docs for detailed specification of geometry and implementation notes
    
    Args
    ----------
    vol_grid : (D,H,W) array
        volume of densities. Linear interpolation of this grid defines the continuous volume.
        
    orientation : (theta_z, theta_y, theta_x) rotation angle in degrees of volume
        around z,y,x axes.
        Note : center of rotation is the center of volume 
        Note : the angle is positive for counterclockwise rotation when observer is facing the axis.
        Note : in general there is no unique set of angles describing a 
            3D orientation as the order matters. We define the order by first rotating vol 
            around its x-axis, then its y-axis, and finally its z-axis.
        Note : theta_y is the 'standard' angle in computed tomography,
            commonly known as the "tilt angle" and theta_z, theta_x are often assumed to be 0.
        Note : Make sure that theta_y, theta_x are kept below 90 degrees in magnitude, as the slope of bounding 
            parallelpiped used to set inegration bounds -> infinity as theta_y,theta_x -> 90.
            
    shift : (y,x) shift of volume in units of ccd pixels. Applied after rotation
    
    voxel_size : dimensions (z,y,x) of a voxel relative to (square) ccd pixel dimensions. 
        (10.0,1.0,1.0) indicates a voxel whose depth is 10x larger than ccd pixel size, 
        and whose height/width are the same as output pixel height/width
    
    ccd_size : (H,W) array, size in number of pixels of returned projection.
        does not have to match (H,W) of input volume.
        Center of ccd is aligned to center of vol (before shifting vol)
        
    ray_oversample : tuple of integers (nz,ny,nx) determining number of ray-points 
        used to numerically evaluate the integrals
        nz : number of ray points per unit length in the perpindicular direction (z) to the CCD
        ny : number of rays per per unit length in the height (y) direction of the CCD
        nx : number of rays per per unit length in the width (x) direction of the CCD

    Returns
    ----------
    projection: (H_CCD,W_CCD) array
    
    Notes
    ---------
    for efficiency, D is expected to be much smaller than H,W
    
    Example
    ----------
    # Project a uniform density volume at a tilt angle of 15 degrees
    >>> import jax.numpy as jnp
    >>> vol_grid = jnp.ones((16,128,128))
    >>> displacement_grid = jnp.zeros((160,160,2))
    >>> proj = project(vol_grid, displacement_grid, orientation=(0.0,15.0,0.0), ccd_size=(160,160))
    >>> print(proj.shape)
    Output: (160, 160)
    """
    # convert to jax arrays
    orientation = jnp.array(orientation)
    shift = jnp.array([0.0,shift[0],shift[1]])

    # create continuous volume and align the volume so its center is at the CCD center
    vol = interpolate_grid(vol_grid)
    vol = scale_img(vol,voxel_size)
    
    mu = jnp.array([0.0, ccd_size[0]/2.0, ccd_size[1]/2.0])
    mu_vol = jnp.array([n*m/2.0 for n,m in zip(vol_grid.shape, voxel_size)])
    delta = mu - mu_vol
    vol = shift_img(vol, delta)
    
    # rotate and shift volume
    vol = rotate_img(vol, mu, orientation)
    vol = shift_img(vol, shift)
    
    # compute integration bounds
    R_inv = jnp.linalg.inv(rot_mat(orientation).transpose())
    b = (shift-mu)@R_inv.transpose() + mu
    z_min = lambda u: (-vol_grid.shape[0]*voxel_size[0]/2 - b[0] - R_inv[0,1] * u[...,0] - R_inv[0,2] * u[...,1]) / R_inv[0,0]
    z_max = lambda u: (+vol_grid.shape[0]*voxel_size[0]/2 - b[0] - R_inv[0,1] * u[...,0] - R_inv[0,2] * u[...,1]) / R_inv[0,0]

    # integrate vol in the vertical direction and exponentiate
    proj = integrate_z(vol, z_min, z_max, n_points=int(ray_oversample[0]*vol_grid.shape[0]*voxel_size[0]))
    proj = map_img(proj, lambda f: jnp.exp(-f))

    # warp the projection
    dy = interpolate_grid(displacement_grid[...,0])
    dx = interpolate_grid(displacement_grid[...,1])
    proj = warp_img2d(proj, dy, dx)
    
    # integrate the projection into pixels
    pixels = pixelate_img(proj, center=mu[1:], grid_size=ccd_size, oversample=ray_oversample[1:])
    
    # return
    return pixels