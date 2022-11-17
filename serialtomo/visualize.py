import plotly.graph_objects as go
import numpy as np
from skimage.transform import downscale_local_mean
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import stackview

from skimage.transform import downscale_local_mean
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection
    
#######################
# stack visualization #
#######################
def viewstack(*stacks, pmin=1.0, pmax=99.0, size=1, rescale=False, view='xy'):
    """ Visualize 3D stacks in Jupyter Notebook
    
    Args
        stacks: depth x height x width. If two stacks are provided, overlay with a curtain
        pmin: float in [0.0,100.0). clip small values in stack by this percentile
        pmax: float in (0.0,100.0]. clip large values in stack by this percentile
        size: float, resize images in xy
        view: 'xy' or 'yz' or 'xz', which axes to display
        
    Returns
        jupyter widget
    """
        
    low, high = pmin, pmax
    if len(stacks) == 1:
        return stackpicker(stacks[0], low, high, size, rescale, view)
    elif len(stacks) == 2:
        return stackcurtain(*stacks, low, high, size, rescale, view)
    else:
        raise ValueError('viewstack only accepts 1 or 2 stacks')

def stackpicker(stack, low=1.0, high=99.0, size=1, rescale=False, view='xy'):
    # convert to numpy
    stack = np.array(stack)
    if stack.ndim == 2:
        stack = stack[None]
    
    # checks
    assert(low >= 0.0 and low <= 100.0)
    assert(high >= 0.0 and high <= 100.0)
    assert(low < high)
    
    # normalize evey section
    if rescale:
        stack = (stack - stack.mean(axis=(1,2),keepdims=True)) / stack.std(axis=(1,2),keepdims=True).clip(1e-16)
    
    # clip
    stack = pclip(stack, low, high)
    
    # permute
    stack = swap_axes(stack, view)
    
    # resize
    stack = resize(stack, size)

    # show
    return stackview.picker(stack,continuous_update=True)

def stackcurtain(stack1, stack2, low=1.0, high=99.0, size=1, rescale=True, view='xy'):
    # convert to numpy
    stack1 = np.array(stack1)
    stack2 = np.array(stack2)
    
    if stack1.ndim == 2 and stack2.ndim == 2:
        stack1 = stack1[None]
        stack2 = stack2[None]
    
    # checks
    assert(low >= 0.0 and low <= 100.0)
    assert(high >= 0.0 and high <= 100.0)
    assert(low < high)
    
    # clip
    stack1 = pclip(stack1, low, high)
    stack2 = pclip(stack2, low, high)
    
    # permute
    stack1 = swap_axes(stack1, view)
    stack2 = swap_axes(stack2, view)
    
    # rescale
    if rescale:
        stack1 = (stack1 - stack1.min()) / (stack1.max() - stack1.min())
        stack2 = (stack2 - stack2.min()) / (stack2.max() - stack2.min())
        
    # resize
    stack1 = resize(stack1, size)
    stack2 = resize(stack2, size)

    # show
    return stackview.curtain(stack1, stack2, continuous_update=True)

#############
# 3d viewer #
#############
def viewstack3d(volume: 'DxHxW array', width=1500, height=1000, low=1.0, high=99.0, size=1):
    # checks
    assert(low >= 0.0 and low <= 100.0)
    assert(high >= 0.0 and high <= 100.0)
    assert(low < high)
    
    # convert to numpy
    print('normalizing....')
    volume = np.array(volume)
    
    # resize
    volume = resize(volume, size)
    
    # clip extreme values
    volume = pclip(volume, low, high)
    
    # create surfaces
    print('creating surfaces...')
    bottom = create_surface(volume, 'bottom')
    top    = create_surface(volume, 'top')
    left   = create_surface(volume, 'left')
    right  = create_surface(volume, 'right')
    front  = create_surface(volume, 'front')
    back   = create_surface(volume, 'back')
    
    # add surfaces and format
    print('adding surfaces...')
    fig = go.Figure(data=[bottom, top, left, right, front, back])
    fig.update_layout(width=width, height=height, plot_bgcolor='white', scene_aspectmode='data')
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )
    
    return fig

def create_surface(volume, view='bottom'):
    volume = volume.transpose(2,1,0)
    W,H,D = volume.shape

    if view == 'bottom': # z = 0
        x,y = np.meshgrid(np.arange(W), np.arange(H))
        z = np.zeros(x.shape,dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)
    elif view == 'top': # z = D-1
        x,y = np.meshgrid(np.arange(W), np.arange(H))
        z = D-1 + np.zeros((W,H),dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)

    elif view == 'left': # x = 0
        y,z = np.meshgrid(np.arange(H), np.arange(D), indexing='ij')
        x = np.zeros((H,D),dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)
    elif view == 'right': # x = W-1
        y,z = np.meshgrid(np.arange(H), np.arange(D), indexing='ij')
        x = W-1 + np.zeros((H,D),dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)

    elif view == 'front': # y = 0
        x,z = np.meshgrid(np.arange(W), np.arange(D), indexing='ij')
        y = np.zeros((W,D),dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)
    elif view == 'back': # y = H-1
        x,z = np.meshgrid(np.arange(W), np.arange(D), indexing='ij')
        y = H-1 + np.zeros((W,D),dtype='int')
        surface = go.Surface(x=x,y=y,z=z,surfacecolor=volume[x,y,z],colorscale='gray',showscale=False)

    return surface

#########
# utils #
#########
def resize(stack, size):
    if size == 1:
        pass
    if size > 1:
        assert(type(size) == int)
        stack = np.repeat(np.repeat(stack,size,axis=-2),size,axis=-1)
    if size < 1:
        size = int(1 / size)
        stack = downscale_local_mean(stack, (1, size, size))
    return stack

def swap_axes(stack, axes):
    if axes == 'xy': pass
    elif axes == 'xz': stack = stack.transpose(1,0,2)
    elif axes == 'yz': stack = stack.transpose(2,0,1)
    else: raise ValueError(f'unrecognized axes "{axes}". Only accepted values are "xy", "xz" or "yz".')
    return stack

def pclip(img, low=1.0, high=99.0):
    low_ = np.percentile(img, low)
    high_ = np.percentile(img, high)
    img = np.clip(img, low_, high_)
    return img
    