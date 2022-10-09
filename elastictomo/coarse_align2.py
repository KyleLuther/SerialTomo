""" Affine regisgration and utilities for working with sequences of affine matrices """
import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import jax
from jax import jit
from flax.linen import avg_pool

import time
from tqdm import tqdm
from pystackreg import StackReg
from skimage.transform import warp, downscale_local_mean

################
# Registration #
################
def register_transform(stack: 'DxHxW', ref_idx=None, downsample=1, verbose=1):
    """ Affine alignment of a stack
    Returns:
        aligned: DxHxW
        A: Dx2x3 matrices that map section k to ref_idx
    """
    t0 = time.time()
    # checks
    assert(stack.ndim == 3)
    assert(stack.dtype == 'uint8') # warp only works with uint8 images
    if ref_idx is None: ref_idx = stack.shape[0] // 2
    
    # downsample stack
    if verbose: print(f'downsampling stack by {downsample}x...')
    pre = downscale_local_mean(stack, (1, downsample, downsample))
    
    # normalize stack
    if verbose: print(f'normalizing {stack.shape} stack...')
    pre = pre.astype('float32')
    pre = pre - pre.mean(axis=(1,2),keepdims=True)
    pre = pre / pre.std(axis=(1,2),keepdims=True)
    
    # compute relative affine alignment
    if verbose: print('registering stack...')
    sr = StackReg(StackReg.AFFINE)
    tmats = sr.register_stack(pre, reference='previous', verbose=verbose)
    
    # transform affine matrices
    if verbose: print('transforming matrices...')
    tmats = np.linalg.inv(tmats)
    tmats = change_reference(square2rect(tmats), ref_idx)
    tmats = square2rect(np.linalg.inv(rect2square(tmats)))
    tmats = swapxy(tmats)
    tmats[...,2] *= downsample
    
    # transform stack
    if verbose: print('transforming stack...')
    aligned = np.zeros(stack.shape, dtype=pre.dtype)
    for i in tqdm(range(stack.shape[0])):
        aligned[i] = warp(stack[i], rect2square(swapxy(tmats[i])), order=1)

    # return
    if verbose: print(f'completed alignment in {time.time()-t0:.3f} sec!')
    return aligned, tmats

# def transform(stack: 'DxHxW', tmats: 'Dx2x3', verbose=1):
#     """ Affine transformation of a stack
#     Returns:
#         aligned: DxHxW
#     """
#     t0 = time.time()
#     # checks
#     assert(stack.ndim == 3)
#     assert((tmats.ndim == 3) and (stack.shape[0] == tmats.shape[0]) and (tmats.shape[1] == 2) and (tmats.shape[2] == 3))
    
#     # transform stack
#     if verbose: print('transforming stack...')
#     aligned = np.zeros(stack.shape, dtype=pre.dtype)
#     for i in tqdm(range(stack.shape[0])):
#         aligned[i] = warp(stack[i], rect2square(swapxy(tmats[i])), order=1)

#     # return
#     return aligned

#########
# Utils #
#########
def swapxy(A: '...x3x3') -> '...x3x3':
    """ flip x and y coordinates in 3x3 transformation matrix """
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

def rect2square(A: '...x2x3') -> '...x3x3':
    """ Convert sequence of 2x3 affines into 3x3 affines """
    assert(A.shape[-2:] == (2,3))
    
    pad = np.zeros(A[...,0:1,:].shape)
    pad[...,-1] = 1.0
    B = np.concatenate([A,pad],axis=-2)
    
    return B

def square2rect(A: '...x3x3') -> '...x2x3':
    """ Convert sequence of 3x3 affines into 2x3 affines """
    assert(A.shape[-2:] == (3,3))
    return A[...,:2,:]

def change_reference(A: 'Kx2x3 array', ref_idx: 'int') -> 'Kx2x3 array':
    """ Convert sequence of absolute affine transforms aligned to first, to be aligned to ref_idx
    Args:
        A: array of 2x3 affine matrices that maps image n to 0
        ref_idx: index that we want to map images to
    Returns:
        B: array of 2x3 affine matrices that maps image n to ref_idx (m)
    """
    A_0N = A
    A_0N = rect2square(A_0N) # Kx3x3
    A_0M = A_0N[ref_idx] # 3x3
    A_MN = np.linalg.inv(A_0M) @ A_0N # Kx3x3
    A_MN = square2rect(A_MN)
    return A_MN

def rel2abs_(A: 'Kx3x3') -> '(K+1)x3x3':
    """ Convert relative to absolute affine matrices """
    assert(A.ndim == 3)
    assert(A.shape[1] == A.shape[2] == 3)
    
    B = [np.eye(A.shape[1])]
    for a in A:
        B.append(B[-1] @ a)
    B = jnp.array(B)
    return B

def rel2abs(A: 'Kx2x3 array', ref_idx: 'int') -> '(K+1)x2x3 array':
    """ Convert sequence of relative affine transforms to absolute affine transforms
    Args:
        A: array of 2x3 affine matrices that maps k+1 to k
    Returns:
        B: array of 2x3 affine matrices, that maps k to ref_idx 
    """
    assert(A.ndim == 3)
    assert(A.shape[1] == 2)
    assert(A.shape[2] == 3)
    assert((ref_idx < A.shape[0]) and (ref_idx >= 0))

    # convert to square matrices
    A = rect2square(A)
    
    # find absolute transforms
    B_plus = rel2abs_(A[ref_idx:])
    B_minus = rel2abs_(np.linalg.inv(A[:ref_idx][::-1]))[1:][::-1]
    B = np.concatenate([B_minus, B_plus])
    
    # convert back to rectangular
    B = B[:,:2,:]
    
    return B

def apply_affine(img: 'HxW', A: '2x3') -> 'HxW':
    """ resample image at coordinates specified by affine transforming coordinates """
    # create grid locations
    H, W = img.shape
    r = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'), axis=-1) + 0.5
    
    # map grid locations
    rp = r @ A[:,:2].transpose(1,0) + A[:,2]
    
    # sample grid locations
    interp = map_coordinates(img, rp.transpose(2,0,1), order=1)
    
    return interp