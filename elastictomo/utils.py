import matplotlib.pyplot as plt
import numpy as np

def plot_vecs(vecs, ax=None, figsize=(6,6), dpi=150, f=1):
    """ Plot vector field """
    assert(len(vecs.shape) == 3 and vecs.shape[2] == 2)
    if ax is None: plt.figure(figsize=figsize,dpi=dpi)
    y = np.arange(0,vecs.shape[0],f)
    x = np.arange(0,vecs.shape[1],f)
    plt.quiver(y,x, vecs[::f,::f,1], -vecs[::f,::f,0])
    plt.gca().set_ylim(plt.gca().get_ylim()[1], plt.gca().get_ylim()[0])
    if ax is None: plt.show()