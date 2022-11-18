[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-abF8gtyAbr59KIZNHtitKbP2HuGbft3?usp=sharing)

## About
**SerialTomo** is a python library designed to aid research in serial section tomography. The forward projection operation is built on top of *Jax*, a high-performance numpy-like library that takes advantage of GPU acceleration in addition to providing automatic differentiation capabilities. The output of ```project``` is a 3D radon transform and is differentiable with respect to volume, tilt angles, and tilt axes.

```python
import jax
import jax.numpy as jnp
from serialtomo import project

# 30 tilts, height = width = 1000 pixels
tilts = jnp.zeros((30,1000,1000))

# sum of squared error between observed tilts and predicted tilts
def loss(volume, tilt_angles, tilt_axis):
    pred = project(volume, tilt_angles, tilt_axis)
    return ((pred-tilts)**2).sum()
    
# function that returns dloss / dvolume
grad_volume = jax.grad(loss, argnums=0)

# function that returns dloss / dtilt_angles
grad_tilt_angles = jax.grad(loss, argnums=1)   

# function that returns dloss / dtilt_axis
grad_tilt_axis = jax.grad(loss, argnums=2)   
```

**SerialTomo** provides other functions which are useful for real-world tomography
-  ```minimize``` Gradient descent with auto learning rate tuning
-  ```alignstacks``` Coarsely aligns multiple adjacent tilt series into a *linogram* representation. Uses SIFT to find correspondences and RANSAC to estimate a projective transformation between pairs.
- ```viewstack``` Interactive scrolling through 3D stacks in a Jupyter Notebook. Built on stack of *stackview*, *ipycanvas* and *ipywidgets*

## Getting started
Check out the colab notebook
- [Simulating and Reconstructing a dual-tilt axes series](https://colab.research.google.com/drive/1-abF8gtyAbr59KIZNHtitKbP2HuGbft3?usp=sharing)
