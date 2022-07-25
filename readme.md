Tomography with Jax 
### High level goal
To make the best EM serial section tomogram anyone has ever seen. We'll acheive this by treating factors such as distortions, illumination times, rotations, shifts as latent variables which we can optimize over to reconstruct the volume. This can be seen as taking [this 2013 paper](https://link.springer.com/chapter/10.1007/978-3-642-38886-6_46) to the extreme. I also see room for improvement via improved interpolation and ray-casting schemes.
<!-- 5 improvements
--------------
- solve for sample warping
- allow for arbitrary constrast per image
- more natural loss function
- improved interpolation strategies
- improved ray sampling strategies -->

<!-- ### Installation

### Example usage

```python
y = project(vol, orientation_params, geometry_params, ray_params)
```
 -->
### Theory
#### The fundamentals
Definitions:
- Define $Y_i$ as the $-\log$ transformed tilt image $i$. So $Y_i(v,u) = \log(255.0) - \log(T_i(v,u))$ where $T_i(v,u)$ is the value of pixel $v,u$ for the $i^{th}$ tilt.

- Define $X$ as the volume we wish to reconstruct. So $X(z,y,x)$ is the value of voxel $z,y,x$.

- Define $X^C$ as the continuous and once differentiable transformation of an image that is generated via linear interpolation. In 1D it is easy to explicitly write this tranform. Let $I$ be a 1D image defined on the discrete grid $r=0.5,1.5,...H-0.5$. We can define the continuous version via
$$ I^C(r) = (r+0.5-\lfloor r+0.5\rfloor) I(\lfloor r+0.5\rfloor) + (r+0.5-\lceil r+0.5\rceil) I(\lceil r+0.5\rceil)$$
We have defined $I(r)=0$ for $r \neq 0.5,1.5,...,H-0.5$

- Define $P_{\theta}$ as the forward projection operator so $P_{\theta} X^C$ generates a tilt image when the volume is oriented at $\theta=\langle \alpha,\beta,\gamma\rangle$ 

Least square minimization of sample densities:
$$ \min_{\mathbf{x}} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{P}_{\alpha_i,\beta,\gamma} \circ X ](v,u) - Y_i(v,u)  \Vert^2 $$

We assume we know the tilt angles at each section and we have performed some heuristic method of alignment (e.g. cross-correlation-based matching)

#### Previous work 
- Variations on schemes that generate the projection $P \circ X$
- Non-negativity constraints on X
- [This 2013 paper](https://link.springer.com/chapter/10.1007/978-3-642-38886-6_46) solved for 3 parameters-per-tilt: the two shift parameters (v,u) and one tilt angle $\alpha$. Define $S_{\Delta}$ as the shift operator so $S_{\Delta} \circ Y^C$ shifts a tilt image by $\Delta=\langle \Delta_v,\Delta_u\rangle$ pixels
$$ [S_{\Delta} \circ Y^C](v,u) := Y(v+\Delta_v,u+\Delta_u)$$
They assumed the other two angles were zero, which worked as they only used simulated data. The optimized the following: 
$$ \min_{\mathbf{x}, v_i, u_i, \alpha_i} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{S}_{v_i,u_i} \circ \mathbf{P}_{\alpha_i,\beta=0,\gamma=0} \circ X ](v,u) - Y_i(v,u)  \Vert^2 $$

#### Future work
- **Solving for all 5 pose parameters** $\Delta_v,\Delta_u,\alpha,\beta,\gamma$
$$ \min_{\mathbf{x} \geq 0, \theta_i, \Delta_i} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{S}_{\Delta} \circ \mathbf{P}_{\theta_i} \circ X ](v,u) - Y_i(v,u)  \Vert^2 $$

- **Distortions** Define $D_{\mathbf{\Delta}}$ as a generic displacement operator so $D_{\mathbf{\Delta}}$ warps a tilt image via a displacement field $\mathbf{\Delta}$
$$ [D_{\mathbf{\Delta}} Y^C](v,u) := Y(v+\Delta_v(v,u),u+\Delta_u(v,u))$$
Note that $\Delta$ is defined on the discrete CCD pixel grid, so we can in principle directly optimize it. Almost certainly, this will require regularization. Also, i don't regularize global shifts. Only deviations for the 5 parameter pose model should be regularized probably:

$$ \min_{\mathbf{x} \geq 0, \theta_i, \Delta_i, \mathbf{\Delta}_i} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{D}_{\Delta_i} \circ \mathbf{S}_{\Delta} \circ \mathbf{P}_{\theta_i} \circ X ](v,u) - Y_i(v,u)  \Vert^2 + \sum_{v=1}^H \sum_{u=1}^W \Vert \mathbf{\Delta}(v,u) \Vert^2 $$

- **Contrast and illumination** Define $a,b$ as the flat-field and contrast offset. We simply add this to every pixel of an image and solve for them too:
$$Y_i(v,u) := -\log(T_i(v,u) + a_i) + b_i$$
$$ \min_{\mathbf{x} \geq 0, \theta_i, \Delta_i, \mathbf{\Delta}_i, a_i, b_i} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{D}_{\Delta_i} \circ \mathbf{S}_{\Delta} \circ \mathbf{P}_{\theta_i} \circ X ](v,u) - \log(T_i(v,u)+a_i) - b_i   \Vert^2 + \sum_{v=1}^H \sum_{u=1}^W \Vert \mathbf{\Delta}(v,u) \Vert^2 $$

- **Loss-function**

- **mean preserving linear interpolation**
- **NERF (neural interpolation)**


*Differentiable images via linear interpolation of finite-resolution images* Basically all I'm going to do is define linear interpolation. Everything is easier if we work with images that satify three properties: 
- it is defined by a finite number of parameters (a regular image satisfies this. # params = # pixels)
- it is defined for floating point pixel coordinates ($I(r)$ exists for all r, not just $r \in 0,1,2,...,W-1$)
- it is a (sub)-differentiable function of the pixel coordinates

It is helpful to define images as continuous and differentiable functions $\mathbb{R}^2arrow\mathbb{R}$. Suppose we are given an image $I$ with pixel values defined on a square grid

Define $I$ as an image. To return the image intensity at pixel location $\mathbf{r} \in \mathbb{R}^2$, we evaluate $I_C(r) = (r-\lfloor r\rfloor) I(\lfloor r\rfloor) + (r-\lfloor r\rceil) I(\lceil r\rceil)$

operators on continuous images. To convert the theory into code, we just bilinearly interpolate images, with the known image values defined on a grid starting at (0.5,0.5). I use a non-standard method to interpolate beyond the borders of the image (that is, when "interpolating" at locations outside the bounding box described by (0.5,0.5) to (img.shape[0]-0.5, img.shape[1]-0.5)) which is described in the docs.

Define $X$ as the volume we wish to solve for. It can be regarded as a continuous function: $\mathbb{R}^3arrow\mathbb{R}$. $X(\mathbf{r})$ is defined as 0 for all $\mathbf{r}$ outside the bounding box of $X$.

Define $\mathbf{P}_{\theta}$ is the operator which takes in a volume $X$ and its orientation $\theta=\langle \alpha,\beta,\gamma\rangle$ and returns a projection of the data.

$T(u)$ is a matrix which warps an image according to the field $\mathbf{u}$. Specifically:
$T(\mathbf{u}) \; \mathbf{r}: \mathbf{r} arrow \mathbf{r} + \mathbf{u}$
$T(\mathbf{u}) \; \mathbf{x} (\mathbf{r}) =  \mathbf{x}(\mathbf{r} + \mathbf{u})$

Interestingly, $P$ and $T$ are linear operators applied to the matrix $x$ if we use linear interpolation schemes. They are nonlinear functions of orientation $\theta$ and generalized displacement fields $u$.

At a high level, to make the best EM serial section tomogram anyone has ever seen. 

Previous work has treated.(essentially latent variables)

We are going to go one step further and treat entire displacement-fields as parameters to be optimized over

**Goal:**
$$ \min_{\mathbf{x}, \theta_i, u_i} \; \sum_{i=1}^N \sum_{v=1}^H \sum_{u=1}^W  \Vert [\mathbf{T}_{u_i} \circ \mathbf{P}_{\theta_i} \circ X ](v,u) - Y_i(v,u)  \Vert^2 $$

### Implemented so far
 - ```project``` Forward projection $P_{\theta} \circ \mathbf{x} $ for arbitrary volume orientations and voxel/ccd geometries, with trilinear interpolation and overcomplete ray sampling.