"""SIFT-based stack projective image registration """
import cv2
import numpy as np
from skimage.transform import downscale_local_mean, warp
from types import SimpleNamespace
from tqdm import tqdm, trange
import sys
import time

#############
# Alignment #
#############
def alignstacks(stacks, ref_idx=None, downsample=10, n_features=3000, lowe_ratio=0.75, verbose=True, sift_kwargs=None, ransac_kwargs=None, tform='homography'):
    """ Aligns stacks with SIFT + RANSAC + projective transformation. 
        
    Args
        stacks: list of [ntilts x height x width arrays] to align
        ref_idx: int, which stack to register to
        downsample: int, factor by which to downsample images before registration to improve speed
        n_features: int, max number of SIFT features per section
        lowe_ratio: float between 0 and 1. Lower means more aggressive filtering of SIFT matches before computing the transform
        verbose: bool, print information while aligning
            
    Returns
        aligned: list of [ntilts x height x width arrays] to align
        info: namespace with alignment info including transformation matrices
        
    Notes
        Registration is performed sequentially within stacks
        Then the central slice of each stack is registered to each other
        The transforms all compose so the stack is aligned to the central section of the central stack
    
    """
    # logging info
    t0 = time.time()
    
    # register
    ref_indices = [s.shape[0]//2 for s in stacks]
    if ref_idx is None: ref_idx = len(ref_indices) // 2
    tmats, tmats_between, tmats_within = register_stacks(stacks, ref_indices, ref_idx, downsample, lowe_ratio, verbose, sift_kwargs, ransac_kwargs, tform)
    
    # transformt
    aligned = transform_stacks(stacks, tmats, verbose)
    
    info = SimpleNamespace()
    info.tmats = tmats
    info.tmats_between = tmats_between
    info.tmats_within = tmats_within

    if verbose:
        tqdm.write(f'completed in {time.time()-t0} sec!', file=sys.stderr)

    return aligned, info


############
# register #
############
def register_stacks(stacks: '[KxHxW]', ref_indices=None, ref_idx=None, downsample=5, lowe_ratio=0.75, verbose=True, sift_kwargs=None, ransac_kwargs=None, tform='homography'):
    """ Projective alignment of a stacks
    Returns:
        H: Kx3x3 array, each 3x3 matrix maps coordinates in image k to ref_idx
        info: useful info
    """
    npairs = sum(s.shape[0] for s in stacks)-1
    with trange(npairs, desc='registering stacks', disable=not verbose, leave=True) as pbar:
        # between stack registration
        if len(stacks) > 1:
            H_between = []
            for i in range(len(stacks)-1):
                H_, info_ = register_pair(stacks[i][ref_indices[i]], stacks[i+1][ref_indices[i+1]], downsample, lowe_ratio, sift_kwargs, ransac_kwargs)
                H_between.append(H_)

                pbar.update()
                pbar.set_postfix({'pair': f'{i,ref_indices[i]}->{i+1,ref_indices[i+1]}', '# candidates': len(info_.accepted_matches), '# matched': info_.accepted_matches.sum()})

            H_between = np.array(H_between)
            H_between = rel2abs(H_between, ref_idx)
        else:
            H_between = [np.eye(3)]

        # within stack registration
        H_within_between = []
        H_within = []
        for i in range(len(stacks)):
            H_within_ = []
            for j in range(len(stacks[i])-1):
                H_, info_ = register_pair(stacks[i][j], stacks[i][j+1], downsample, lowe_ratio, sift_kwargs, ransac_kwargs, tform)
                H_within_.append(H_)

                pbar.update()
                pbar.set_postfix({'pair': f'{i,j}->{i,j+1}', '# candidates': len(info_.accepted_matches), '# matched': info_.accepted_matches.sum()})

            H_within_ = np.array(H_within_)
            H_within_ = rel2abs(H_within_, ref_indices[i])
            H_within.append(H_within_)

            H_within_between_ = np.einsum('ij,tjk->tik',H_between[i],H_within_)
            H_within_between.append(H_within_between_)

    return H_within_between, H_between, H_within

def register_pair(ref: 'HxW uint8', mov: 'HxW uint8', downsample=10, lowe_ratio=0.75, sift_kwargs=None, ransac_kwargs=None, tform='homography'):
    """ Projective registration of pair of images using SIFT features
    
    Args
    
    Returns
        H: 3x3 homography matrix: maps mov coordinates to ref, ref(r) = mov(Hr)
        info: namespace containing useful info
    """
    # checks
    assert((ref.ndim == 2) and (mov.ndim == 2))
    assert((lowe_ratio > 0.0) and (lowe_ratio <= 1.0))
    
    # downsample for speed
    ref = downscale_local_mean(ref, (downsample,downsample))
    mov = downscale_local_mean(mov, (downsample,downsample))
    
    # normalize images
    im1 = normalize_img(mov)
    im2 = normalize_img(ref)
    
    # detect and compute descriptors
    if sift_kwargs is None: sift_kwargs = {}
    fts = cv2.SIFT_create(**sift_kwargs)
    # fts = cv2.ORB_create(**sift_kwargs)
    
    kp1, des1 = fts.detectAndCompute(im1, None)
    kp2, des2 = fts.detectAndCompute(im2, None)
    
    # match features
    if len(kp1) < 4 or len(kp2) < 4:
        H = np.eye(3)
        info = SimpleNamespace()
        info.accepted_matches = np.array([])
        return H, info
        
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

    try:
        if ransac_kwargs is None: ransac_kwargs = {}
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, **ransac_kwargs)
        if tform == 'homography':
            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, **ransac_kwargs)
        elif tform == 'affine':
            H, mask = cv2.estimateAffine2D(points1, points2, cv2.RANSAC, **ransac_kwargs)
            H = np.concatenate([H,np.array([0.0,0.0,1.0])[None]])
        else:
            raise ValueError(f'Unrecognized tform: {tform}')
        mask = mask[:,0]
    except:
        H = np.eye(3)
        mask = np.zeros(points1.shape[0],dtype='uint8')
        
    # correct for downsampling
    H[:2,2] *= downsample
    H[2,:2] /= downsample
    
    # logging info
    info = SimpleNamespace()
    info.key_points1 = points1
    info.key_points2 = points2
    info.accepted_matches = mask
    
    return H, info

#################
# Transforming #
#################
def transform_stacks(stacks: 'KxHxW', H: 'Kx3x3', verbose=True) -> 'KxHwW':
    nimages = sum(s.shape[0] for s in stacks)
    with trange(nimages, desc='transforming stacks', disable=not verbose, leave=True) as pbar:
        outs = []
        for stack, h in zip(stacks,H):
            out = np.zeros(stack.shape, dtype=stack.dtype)
            for i in range(stack.shape[0]):
                if stack[i].dtype == np.float32: # CV2 doesn't work with float32
                    img = stack[i].astype(np.float64)
                    out[i] = cv2.warpPerspective(img, h[i], stack[i].shape[::-1], flags=1, borderValue = img.mean()) # linear interpolation
                else:
                    out[i] = cv2.warpPerspective(stack[i], h[i], stack[i].shape[::-1], flags=1, borderValue = stack[i].mean()) # linear interpolation
                pbar.update()
            outs.append(out)
        
    return outs

#################
# Preprocessing #
#################
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

def pclip(img, low=1.0, high=99.0):
    low_ = np.percentile(img, low)
    high_ = np.percentile(img, high)
    img = np.clip(img, low_, high_)
    return img

def to_uint8(img, low=0.1, high=99.9, rescale=True):
    img = pclip(img, low, high)
    if rescale:
        img = img - img.min()
        img = 255. * img / img.max()
    img = img.astype('uint8')
    return img

#########
# Utils #
#########
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
