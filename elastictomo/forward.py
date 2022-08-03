""" Code to simulate forward projection """
import jax 
import jax.numpy as jnp
import numpy as np
from elastictomo.interpolate import pixelate

# helper functions
def rot_mat(theta):
    """ Creates Euler rotation matrix
    Args :
        theta : (tz,ty,tz) in degrees
    Returns:
        R : (3,3) matrix equal to R_y @ R_z @ R_x
    """
    theta_z, theta_y, theta_x = tuple(-o*np.pi/180 for o in theta)
    Rx_ = jnp.array([[1,0,0],[0,jnp.cos(theta_x),-jnp.sin(theta_x)],[0,jnp.sin(theta_x),jnp.cos(theta_x)]])
    Rz_ = jnp.array([[jnp.cos(theta_z),-jnp.sin(theta_z),0],[jnp.sin(theta_z),jnp.cos(theta_z),0],[0,0,1]])
    Ry_ = jnp.array([[jnp.cos(theta_y),0,jnp.sin(theta_y)],[0,1,0],[-jnp.sin(theta_y),0,jnp.cos(theta_y)]])
    R_ = Ry_ @ Rz_ @ Rx_
    R = jnp.flip(R_,axis=(0,1))
    return R

def integrate_z(img, z_min, z_max, n_points):
    """ Integrates 3D image along z axis
    Args : 
        img : R^3 -> R, continuous image 
        z_min : R^2 -> R, z_min(y,x) is the lower bound function
        z_max : R^2 -> R, z_max(y,x) is the upper bound function
        n_points : int,  number of points to use to numerically integrate
    Returns :
        integral : R^2 -> R, continuous image giving numerical integral
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

# simulate propagation through a volume with lens warping
def forward(vol: "f: R^3 -> R", disp: "f: R^2 -> R^2", vol_size: "(D,H,W)", ccd_size: "(H,W)",
            orientation: "(theta_z,theta_y,theta_x)", energy=1.0, oversample=(1,1,1)):
    """ Simulate electron transmission and image acquisition through 3d volume
    
    Refer to docs for detailed specification of geometry and implementation notes
    
    Args
    ----------
    vol : volume of densities. 
        Maps continuous coordinates [z \in (0,D), y \in (0,H), x \in (0,W)] to density(z,y,x).
        center of volume is initially aligned to microscope origin (0,0,0)
        
    disp : vector field of displacements applied after radon transform. 
        Maps coordinates in CCD frame (y,x) to displacement(y,x)
        
    vol_size : (D,H,W) where D,H,W are sizes relative to (square) CCD pixels.
        CCD origin (0,0,0) assumed to be in center of volume.
    
    ccd_size : (H,W) array, size in number of pixels of returned projection.
        does not have to match (H,W) of input volume.
        CCD origin (0,0) assumed to be in center of CCD   
        
    orientation : (theta_z, theta_y, theta_x) rotation angle in degrees of volume
        around z,y,x axes.
        Note : center of rotation is the microscope origin (0,0,0)
        Note : the angle is positive for counterclockwise rotation when observer is facing the axis.
        Note : in general there is no unique set of angles describing a 
            3D orientation as the order matters. We define the order by first rotating vol 
            around its x-axis, then its y-axis, and finally its z-axis.
        Note : theta_y is the 'standard' angle in computed tomography,
            commonly known as the "tilt angle" and theta_z, theta_x are often assumed to be 0.
        Note : Make sure that theta_y, theta_x are kept below 90 degrees in magnitude, as the slope of bounding 
            parallelpiped used to set inegration bounds -> infinity as theta_y,theta_x -> 90.
                
    oversample : tuple of integers (nz,ny,nx) determining number of ray-points 
        used to numerically evaluate the integrals
        nz : number of ray points per unit length in the perpindicular direction (z) to the CCD
        ny : number of rays per per unit length in the height (y) direction of the CCD
        nx : number of rays per per unit length in the width (x) direction of the CCD

    Returns
    ----------
    projection: (H, W) array
    
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
    # center volume at detector origin
    centered_vol = lambda r: vol(r + jnp.array(vol_size) / 2)
    
    # rotate volume around the microscope origin
    rotated = lambda r: centered_vol(r @ rot_mat(orientation).transpose())
    
    # compute integration bounds
    R_inv = jnp.linalg.inv(rot_mat(orientation).transpose())
    z_min = lambda u: (-vol_size[0]/2 - R_inv[0,1] * u[...,0] - R_inv[0,2] * u[...,1]) / R_inv[0,0]
    z_max = lambda u: (+vol_size[0]/2 - R_inv[0,1] * u[...,0] - R_inv[0,2] * u[...,1]) / R_inv[0,0]

    # integrate vol in the vertical direction and exponentiate
    proj = integrate_z(rotated, z_min, z_max, n_points=int(oversample[0]*vol_size[0]))
    exp = lambda r: energy * jnp.exp(-proj(r))

    # warp the projection
    warped = lambda r: exp(r - disp(r+jnp.array(ccd_size)/2))
    
    # integrate the projection into pixels
    centered = lambda r: warped(r-jnp.array(ccd_size)/2)
    pixels = pixelate(centered, ccd_size, oversample=oversample[1:])
    
    # return
    return pixels