import jax 
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from flax.linen import avg_pool
import numpy as np

def project(vol: '(D,H,W) array', orientation=(0.0,0.0,0.0), rotation_center=None,
            voxel_size=(1.0,1.0,1.0), ccd_size=None, ray_oversample=(1,1,1)):
    """ Compute a forward projection via ray tracing. 
    
    Refer to docs for detailed specification of geometry and implementation notes
    
    Args
    ----------
    vol : (D,H,W) array
        volume of densities to project to ccd grid
        
    orientation : counterclockwise rotation angles (alpha,beta,gamma) in degrees.
        alpha : rotation around specimen's y-axis ("tilt angle")
        beta : rotation around specimen's z-axis ("tilt rotation")
        gamma : rotation around specimen's x-axis ("beam tilt")
        
    rotation_center : location (z,y,x) in volume coordinate frame of the center of rotation.
        The center of rotation is used to orient the volume relative to CCD.
        Center of rotation is always located vertically above the center of
        the CCD output. If None, then the center of vol (D/2,H/2,W/2) is used
        
    voxel_size : dimensions (z,y,x) of a voxel relative to (square) ccd pixel dimensions. 
        (10.0,1.0,1.0) indicates a voxel whose depth is 10x 
        larger than ccd pixel size, and whose height/width
        are the same as output pixel height/width
        
    ccd_size : (H,W) array
        size in number of pixels of returned projection.
        does not have to match (H,W) of input volume
        If None, uses number of CCD pixels needed to cover volume at a tilt of 0,0,0
        
    ray_oversample : tuple of integers (fw, fv, fu) determining number of ray-points 
        used to compute the line integral that defines forward projection
        fw : number of points per ray in the perpindicular direction (w) to the CCD
        fv : number of rays per pixel in the height (v) direction of the CCD
        fu : number of rays per pixel in the width (u) direction of the CCD

    
    Returns
    ----------
    projection: (H,W) array
    
    Example
    ----------
    # Project a uniform density volume at a tilt angle of 15 degrees
    >>> import jax.numpy as jnp
    >>> vol = jnp.ones((16,128,128))
    >>> projection = project(vol, orientation=(15.0,0.0,0.0), ccd_size=(160,160))
    >>> print(projection.shape)
    Output: (160, 160)
    
    """    
    # 0: setting shapes and geometry
    if rotation_center is None:
        rotation_center = tuple(s/2 for s in vol.shape)
    if ccd_size is None:
        ccd_size = (int(np.ceil(vol.shape[1]*voxel_size[1])),int(np.ceil(vol.shape[2]*voxel_size[2])))
        
    # 1: compute ray start & end points in detector (v,u) coordinate frame
    v = jnp.linspace(0.5/ray_oversample[1], ccd_size[0]-0.5/ray_oversample[1], ccd_size[0]*ray_oversample[1])
    u = jnp.linspace(0.5/ray_oversample[2], ccd_size[1]-0.5/ray_oversample[2], ccd_size[1]*ray_oversample[2])
    u,v = jnp.meshgrid(u,v)
    start_z = jnp.zeros_like(v) + 0.5/ray_oversample[0]
    end_z = vol.shape[0]*jnp.ones_like(v)-0.5/ray_oversample[0]

    # 2: transform ray start/end points (z,v,u) into voxel (z,y,x) coordinate frame
    # RS(x_v - mu_v) = x_c - mu_c
    # x_v = A x_c + b where A = S_inv R^T and b = -A mu_c + mu_v
    mu_v = jnp.array(rotation_center)
    mu_c = jnp.array([0,ccd_size[0]/2.,ccd_size[1]/2.])

    # rotation 
    alpha,beta,gamma = tuple(o*np.pi/180 for o in orientation)
    Rx_ = jnp.array([[1,0,0],[0,jnp.cos(gamma),-jnp.sin(gamma)],[0,jnp.sin(gamma),jnp.cos(gamma)]])
    Rz_ = jnp.array([[jnp.cos(beta),-jnp.sin(beta),0],[jnp.sin(beta),jnp.cos(beta),0],[0,0,1]])
    Ry_ = jnp.array([[jnp.cos(alpha),0,jnp.sin(alpha)],[0,1,0],[-jnp.sin(alpha),0,jnp.cos(alpha)]])
    R_ = Ry_ @ Rz_ @ Rx_
    # R_ = Rx_ @ Rz_ @ Ry_
    R = jnp.flip(R_,axis=(0,1)) # recall, we are always doing zyx coordinates

    # scaling
    S_inv = jnp.diag(1 / jnp.array(voxel_size))

    # generate transform matrix
    A = S_inv @ R#.transpose()
    b = - A @ mu_c + mu_v

    M1 = jnp.zeros((3,3)).at[0,0].set(1.0)
    M2 = jnp.zeros((3,3)).at[1,1].set(1.0)
    M3 = jnp.zeros((3,3)).at[2,2].set(1.0)

    C = (-M1 + A @ M2 + A @ M3)
    D = (-A @ M1 + M2 + M3)
    D_inv = jnp.linalg.inv(D)
    E = D_inv @ C
    f = D_inv @ b

    # transform
    starts_zvu = jnp.stack([start_z,v,u],axis=-1)
    ends_zvu = jnp.stack([end_z,v,u],axis=-1)

    starts_wyx = starts_zvu @ E.transpose() + f[...,:]
    ends_wyx = ends_zvu @ E.transpose() + f[...,:]

    starts_zyx = jnp.stack([start_z,starts_wyx[...,1],starts_wyx[...,2]],axis=-1)
    ends_zyx = jnp.stack([end_z,ends_wyx[...,1],ends_wyx[...,2]],axis=-1)
    
    # 3: create points along ray (qd x ph x pw x 3)
    rays_zyx = ends_zyx-starts_zyx
    ray_lens = jnp.linalg.norm(rays_zyx,axis=-1)
    seg_lens = (ray_lens + 1 / ray_oversample[0]) / (vol.shape[0] * ray_oversample[0])
    ray_dirs_zyx = rays_zyx / ray_lens[...,None]
    points = starts_zyx[None,...] + ray_dirs_zyx[None,...] * jnp.linspace(0,ray_lens,vol.shape[0]*ray_oversample[0])[...,None]
    
    # return points

    # 4: evaluate image along points
    # map_coordinates returns nonzero values outside the image boundaries
    # so i'll just zero-out everywhere outside the image
    mask = jnp.ones_like(points[...,0])
    mask = mask.at[(points[:,:,:,0] <= 0.0) | (points[:,:,:,0] >= vol.shape[0])].set(0.0)
    mask = mask.at[(points[:,:,:,1] <= 0.0) | (points[:,:,:,1] >= vol.shape[1])].set(0.0)
    mask = mask.at[(points[:,:,:,2] <= 0.0) | (points[:,:,:,2] >= vol.shape[2])].set(0.0)
    
    # pad one pixel so linear interpolation makes sense on the boundaries
    vol = jnp.pad(vol,1,mode='reflect') 
    points = points + 1

    # map_coordinates places coordinate centers at 0, not 0.5
    points = points.transpose(3,0,1,2)-0.5
    densities = mask*map_coordinates(vol,points,order=1)
    
    # 5: integrate along points
    ray_sums = (densities*seg_lens).sum(0)

    # 6: sample to ccd values
    projection = avg_pool(ray_sums[None,...,None],
                          window_shape=(ray_oversample[1],ray_oversample[2]),
                          strides=(ray_oversample[1],ray_oversample[2]))[0,...,0]
    
    return projection
