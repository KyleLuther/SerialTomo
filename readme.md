**SerialTomo** is a python library designed to aid research in serial section tomography. The key functionality is built on top of *Jax*, a high-performance numpy-like library that takes advantage of GPU acceleration in addition to providing automatic differentiation capabilities.

**SerialTomo** provides 4 key functions
- ```project``` Applies a radon transform to a 3D volume. The output is differentiable with respect to volume, tilt angles, and tilt axes
-  ```alignstacks``` Coarsely aligns multiple adjacent tilt series into a *linogram* representation. Uses SIFT to find correspondences and RANSAC to estimate a projective transformation between pairs.
-  ```minimize``` Gradient descent with auto learning rate tuning
- ```viewstack``` Interactive scrolling through 3D stacks in a Jupyter Notebook. Built on stack of *stackview*, *ipycanvas* and *ipywidgets*
