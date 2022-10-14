import numpy as np
import matplotlib.pyplot as plt
import stackview
from tqdm import tqdm

from skimage.transform import downscale_local_mean
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection
    
#######################
# stack visualization #
#######################
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
    else: raise ValueError(f'unrecognized axes order {axes}')
    return stack

def stackpicker(stack, low=1.0, high=99.0, size=1, norm_every=False, axes='xy'):
    # convert to numpy
    stack = np.array(stack)
    
    # checks
    assert(low >= 0.0 and low <= 100.0)
    assert(high >= 0.0 and high <= 100.0)
    assert(low < high)
    
    # normalize evey section
    if norm_every:
        stack = (stack - stack.mean(axis=(1,2),keepdims=True)) / stack.std(axis=(1,2),keepdims=True).clip(1e-16)
    
    # clip
    stack = pclip(stack, low, high)
    
    # permute
    stack = swap_axes(stack, axes)
    
    # resize
    stack = resize(stack, size)

    # show
    return stackview.picker(stack,continuous_update=True)

def stackcurtain(stack1, stack2, low=1.0, high=99.0, size=1, rescale=True, axes='xy'):
    # convert to numpy
    stack1 = np.array(stack1)
    stack2 = np.array(stack2)
    
    # checks
    assert(low >= 0.0 and low <= 100.0)
    assert(high >= 0.0 and high <= 100.0)
    assert(low < high)
    
    # clip
    stack1 = pclip(stack1, low, high)
    stack2 = pclip(stack2, low, high)
    
    # permute
    stack1 = swap_axes(stack1, axes)
    stack2 = swap_axes(stack2, axes)
    
    # rescale
    if rescale:
        stack1 = (stack1 - stack1.min()) / (stack1.max() - stack1.min())
        stack2 = (stack2 - stack2.min()) / (stack2.max() - stack2.min())
        
    # resize
    stack1 = resize(stack1, size)
    stack2 = resize(stack2, size)

    # show
    return stackview.curtain(stack1, stack2, continuous_update=True)

##############################
# displacement visualization #
##############################
def plotvecs(vecs, ax=None, figsize=(6,6), dpi=150, f=1):
    """ Plot vector field """
    assert(len(vecs.shape) == 3 and vecs.shape[2] == 2)
    if ax is None: plt.figure(figsize=figsize,dpi=dpi)
    y = np.arange(0,vecs.shape[0],f)
    x = np.arange(0,vecs.shape[1],f)
    plt.quiver(y,x, vecs[::f,::f,1], -vecs[::f,::f,0])
    plt.gca().set_ylim(plt.gca().get_ylim()[1], plt.gca().get_ylim()[0])
    if ax is None: plt.show()
    
def draw_displacement_grid(displacements: 'NxMx2', grid_size=1, **kwargs):
    """ Draws grid defined by displacement grid """    
    # compute locations
    N,M = displacements.shape[:2]
    r = np.stack(np.meshgrid(np.arange(N), np.arange(M), indexing='ij'), axis=-1)
    r = r * grid_size
    locations = r+displacements
    
    # add line segments
    ax = plt.gca()
    segs1 = locations
    ax.add_collection(LineCollection(segs1, **kwargs))

    segs2 = locations.transpose(1,0,2)
    if 'label' in kwargs:
        label = kwargs.pop('label')
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    
def fig2img(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype('int32')
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return img
    
def overlay_displacement_grid_(img: 'HxW', displacements: 'NxMx2', grid_size=256, line_width=300, **kwargs):
    """ Returns an rgb image with displacements plotted on top of image """
    h,w = img.shape
    dpi = 1
    
    # create figure
    fig = plt.figure(figsize=(w,h), dpi=dpi)
    plt.axes((0,0,1,1)) 
    
    # add image
    plt.imshow(img, aspect='auto', cmap='gray')
    
    # add grid
    draw_displacement_grid(displacements, grid_size, **kwargs)

    # get image
    plt.gca().set_axis_off()
    img = fig2img(fig)
    plt.close()
    
    return img

def overlay_displacement_grid(imgs: 'KxHxW', displacements: 'KxNxMx2', grid_size=256, line_width=300, **kwargs):
    out = []
    for img, dsp in tqdm(zip(imgs, displacements)):
    # for i in tqdm(range(imgs.shape[0])):
        # img, dsp = imgs[i], displacements[i]
        out.append(overlay_displacement_grid_(img, dsp, grid_size, line_width, **kwargs))
    return np.array(out)

# plotting utils
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# general visualization utilts
def pclip(img, low=1.0, high=99.0):
    low_ = np.percentile(img, low)
    high_ = np.percentile(img, high)
    img = np.clip(img, low_, high_)
    return img

# image conversion
def to_uint8(img, low=0.1, high=99.9, rescale=True):
    img = pclip(img, low, high)
    if rescale:
        img = img - img.min()
        img = 255. * img / img.max()
    img = img.astype('uint8')
    return img