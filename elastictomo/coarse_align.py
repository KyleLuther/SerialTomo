"""SIFT-based stack projective image registration """
import cv2
import numpy as np
from skimage.transform import downscale_local_mean, warp
from types import SimpleNamespace
from tqdm import tqdm, trange
import sys

from elastictomo.utils import pclip

#########
# Notes #
#########
# cv2 uses center of the upper left pixel as 0,0

###########################################
# Alignment: Preprocess-Register-Transform #
###########################################
def align_stack(stack: 'KxHxW uint8', downsample=10, ref_idx=None, n_features=3000, lowe_ratio=0.75, method='sequential', clip_percentile=1.0, **kwargs):
    """ aligns stack """
    assert((stack.ndim ==3))
    if ref_idx is None: ref_idx = stack.shape[0] // 2
    
    # logging info
    tqdm.write(f'performing {method}-based alignment of {stack.shape} {stack.dtype} stack to section {ref_idx}...', file=sys.stderr)
    # print(f'performing {method}-based alignment of {stack.shape} stack to section {ref_idx}...')
    
    # preprocess
    pre = downsample_stack(stack, downsample)
    pre = normalize_stack(pre, clip_percentile)
    
    # register
    H, info = register_stack(pre, ref_idx, n_features, lowe_ratio, method, **kwargs)
    H[:,:2,2] *= downsample
    H[:,2,:2] /= downsample
    
    # transform
    aligned = transform_stack(stack, H)

    # return
    info.H = H
    return aligned, info

##################
# Transformation #
##################
def transform_stack(stack: 'KxHxW', H: 'Kx3x3', verbose=True) -> 'KxHwW':
    assert(len(stack) == len(H))
    assert((stack.ndim==3) and (H.ndim==3) and (H.shape[1] == H.shape[2] == 3))
    
    out = np.zeros(stack.shape, dtype=stack.dtype)
    for i in trange(stack.shape[0], desc='transforming', disable=not verbose):
        out[i] = cv2.warpPerspective(stack[i], H[i], stack[i].shape[::-1], flags=1, borderValue = stack[i].mean()) # linear interpolation
        
    return out

################
# Registration #
################
def register_stack(stack: 'KxHxW uint8', ref_idx=None, n_features=3000, lowe_ratio=0.75, method='sequential', verbose=True, **kwargs):
    """ Projective alignment of a stack
    Returns:
        H: Kx3x3 array, each 3x3 matrix maps coordinates in image k to ref_idx
        info: useful info
    """
    # checks
    assert(stack.ndim == 3)
    assert(stack.dtype == 'uint8') # for register pair
    if ref_idx is None: ref_idx = stack.shape[0] // 2
    
    # compute transformation matrices
    if method == 'sequential':
        H, infos = [], []
        for i in (pbar := trange(stack.shape[0]-1, desc='registering with SIFT', disable=not verbose)):
            H_, info_ = register_pair(stack[i], stack[i+1], n_features, lowe_ratio, **kwargs)
            H.append(H_)
            infos.append(info_)
            pbar.set_postfix({'# candidates': len(info_.accepted_matches), '# matched': info_.accepted_matches.sum()})
        H = np.array(H)
        H = rel2abs(H, ref_idx)
        
    elif method == 'reference':
        H, infos = [], []
        for i in (pbar := trange(stack.shape[0], desc='registering with SIFT', disable=not verbose)):
            H_, info_ = register_pair(stack[ref_idx], stack[i], n_features, lowe_ratio, **kwargs)
            H.append(H_)
            infos.append(info_)
            pbar.set_postfix({'# candidates': len(info_.accepted_matches), '# matched': info_.accepted_matches.sum()})
        H = np.array(H)
    
    else:
        raise ValueError(f'Unrecognized method: {method}')
    
    # reformat info
    info = SimpleNamespace()
    info.key_points1 = [inf.key_points1 for inf in infos]
    info.key_points2 = [inf.key_points2 for inf in infos]
    info.accepted_matches = [inf.accepted_matches for inf in infos]

    return H, info

def register_pair(ref: 'HxW uint8', mov: 'HxW uint8', n_features=3000, lowe_ratio=0.75, **kwargs):
    """ Projective registration of pair of images using SIFT features
    Returns:
        H: 3x3 homography matrix: maps mov coordinates to ref, ref(r) = mov(Hr)
        info: namespace containing useful info
    Notes: 
        Does not perform normalization. It is extremeley sensitive however to details of normalization
        and expects that the inputs have already been normalized
    """
    # checks
    assert((ref.dtype == 'uint8') and (mov.dtype == 'uint8')) # cv2 requires uint8 inputs
    assert((ref.ndim == 2) and (mov.ndim == 2))
    assert((lowe_ratio > 0.0) and (lowe_ratio <= 1.0))
    
    # detect and compute descriptors
    im1, im2 = mov, ref
    fts = cv2.SIFT_create(nfeatures=n_features)
    kp1, des1 = fts.detectAndCompute(im1, None)
    kp2, des2 = fts.detectAndCompute(im2, None)
    
    # match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # filter out non-confident matches via Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            good.append(m)
            
    matches = sorted(good, key = lambda x:x.distance)
    dists = [m.distance for m in matches]
    
    # compute homography (3x3 transform) with RANSAC
    points1 = np.zeros((len(matches), 2), dtype=np.float64)
    points2 = np.zeros((len(matches), 2), dtype=np.float64)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, **kwargs)
    mask = mask[:,0]
    
    # logging info
    info = SimpleNamespace()
    info.key_points1 = points1
    info.key_points2 = points2
    info.accepted_matches = mask
    
    return H, info

#################
# Preprocessing #
#################
def downsample_stack(stack: 'KxHxW', downsample: 'int', verbose=True):
    assert((stack.shape[1]%downsample == 0) and (stack.shape[1]%downsample == 0))
    
    out = np.zeros((stack.shape[0], stack.shape[1]//downsample, stack.shape[2]//downsample), dtype=stack.dtype)
    for i in trange(stack.shape[0], desc=f'downsampling by {downsample}x', disable=not verbose):
        out[i] = downscale_local_mean(stack[i], downsample)
        
    return out

def normalize_img(img: 'HxW array', clip_percentile=1.0) -> 'HxW uint8 array':
    # center and scale every image in the stack
    img = img - img.mean(axis=(-1,-2),keepdims=True)
    img = img  / img.std(axis=(-1,-2),keepdims=True).clip(1e-16)

    # convert to uint8
    img = pclip(img,clip_percentile,100.0-clip_percentile)
    img = img - img.min()
    img = img / img.max().clip(1e-16)
    img = (255.*img).astype('uint8')
    
    return img

def normalize_stack(img: '...xHxW array', clip_percentile=1.0, verbose=True) -> '...xHxW uint8 array':
    out = np.zeros(img.shape,dtype='uint8')
    for i in trange(img.shape[0], desc=f'normalizing', disable=not verbose):
        out[i] = normalize_img(img[i])
    return out

# def normalize_stack(img: '...xHxW array', clip_percentile=1.0) -> '...xHxW uint8 array':
#     # center and scale every image in the stack
#     img = img - img.mean(axis=(-1,-2),keepdims=True)
#     img = img  / img.std(axis=(-1,-2),keepdims=True).clip(1e-16)
    
#     # convert to uint8
#     img = pclip(img,clip_percentile,100.0-clip_percentile)
#     img = img - img.min()
#     img = img / img.max().clip(1e-16)
#     img = (255.*img).astype('uint8')
    
#     return img

#########
# Utils #
#########
def swapxy(A: '...x3x3') -> '...x3x3':
    """ flip x and y coordinates in affine matrix """
    assert(A.shape[-2:] == (3,3))
    
    # flip cols
    B = A.copy()
    B[...,0] = A[...,1]
    B[...,1] = A[...,0]
    
    # flip rows
    C = B.copy()
    C[...,0,:] = B[...,1,:]
    C[...,1,:] = B[...,0,:]
    
    return C

def change_reference(A: 'Kx3x3 array', ref_idx: 'int') -> 'Kx3x3 array':
    """ Convert sequence of absolute transformations aligned to first, to be aligned to ref_idx
    Args:
        A: array of 3x3 transformation matrices that maps image n to 0
        ref_idx: index that we want to map images to
    Returns:
        B: array of 3x3 transformation matrices that maps image n to ref_idx (m)
    """
    A_0N = A
    A_0M = A_0N[ref_idx] # 3x3
    A_MN = np.linalg.inv(A_0M) @ A_0N # Kx3x3
    A_MN = A_MN / A_MN[...,2:3,2:3] # normalize
    return A_MN

def rel2abs_(A: 'Kx3x3') -> '(K+1)x3x3':
    """ Convert relative to absolute affine matrices """
    assert(A.ndim == 3)
    assert(A.shape[1] == A.shape[2] == 3)
    
    B = [np.eye(A.shape[1])]
    for a in A:
        B.append(B[-1] @ a)
    B = np.array(B)
    return B

def rel2abs(A: 'Kx3x3 array', ref_idx: 'int') -> '(K+1)x3x3 array':
    """ Convert sequence of relative affine transforms to absolute affine transforms
    Args:
        A: array of 3x3 affine matrices that maps k+1 to k
    Returns:
        B: array of 3x3 affine matrices, that maps k to ref_idx 
    """
    assert(A.ndim == 3)
    assert(A.shape[-1] == 3)
    assert(A.shape[-2] == 3)
    assert((ref_idx <= A.shape[0]) and (ref_idx >= 0))

    # find absolute transforms
    B_plus = rel2abs_(A[ref_idx:])
    B_minus = rel2abs_(np.linalg.inv(A[:ref_idx][::-1]))[1:][::-1]
    B = np.concatenate([B_minus, B_plus])
    
    # renormalize
    B = B / B[...,2:3,2:3]
    
    return B

def apply_homography(r: '...x2', H: '3x3') -> '...x2':
    r = np.concatenate([r, np.ones(r.shape[:-1])[...,None]], axis=-1) # homogeneous coordinates
    r = r @ H.T # apply homography
    r = r[...,:2] / r[...,2:3] # rescale
    return r

#################
# Visualization #
#################
# todo: visualize the transformtions
# def show_transformation_(img: 'HxW', H: '3x3', downsample=10, padding):
#     # find 
    
    
    
# def show_transformation(stack: 'KxHxW', H: 'Kx3x3', downsample=10, padding):
#     # for each image
#     # find 




###############
# In progress #
###############
# def apply_affine(img: 'HxW', A: '2x3') -> 'HxW':
#     """ resample image at coordinates specified by affine transforming coordinates """
#     # create grid locations
#     H, W = img.shape
#     r = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'), axis=-1) + 0.5
    
#     # map grid locations
#     rp = r @ A[:,:2].transpose(1,0) + A[:,2]
    
#     # sample grid locations
#     interp = map_coordinates(img, rp.transpose(2,0,1), order=1)
    
#     return interp