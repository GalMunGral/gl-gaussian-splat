# WebGPU Gaussian Splatting

**Live demo:** https://galmungral.github.io/gl-gaussian-splat/

## Rhetorical Design

### Purpose

[gl-raytracer](https://github.com/GalMunGral/gl-raytracer) showed that a fragment shader can serve as a general-purpose parallel compute kernel. This project takes that observation further: in a Gaussian splatting renderer, the graphics pipeline contributes almost nothing. The vertex shader reads precomputed positions from a storage buffer; the fragment shader evaluates a 2D Gaussian falloff and blends. The real work — spherical harmonics evaluation, covariance projection, depth sorting — runs entirely in compute shaders. The boundary between rendering and general compute is architectural, not conceptual.

Gaussian splatting also represents a break from conventional scene representations. There is no mesh, no UV map, no artist-authored geometry. The scene is a cloud of semi-transparent ellipsoids fit to photographs — a learned representation. This connects [cpu-raytracer](https://github.com/GalMunGral/cpu-raytracer) and [gl-raytracer](https://github.com/GalMunGral/gl-raytracer), which operate on explicit geometry, to a new paradigm where the representation itself is the output of an optimization.

### Strategy

Dragging the scene dispatches a prepass compute shader and a radix sort compute shader before the render pass executes. The graphics pipeline receives fully prepared data and contributes no logic of its own. For an audience familiar with the rasterization pipeline, this inversion is the point.

## Technical Challenges

### The Gaussian Splatting Model

Each Gaussian is parameterized by a center $\mu \in \mathbb{R}^3$, a covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$ (stored as a lower-triangular factor), degree-3 spherical harmonic coefficients per color channel, and an opacity $\sigma$.

**Projection.** Let $W$ be the rotation part of the view matrix and $J$ the Jacobian of the perspective divide evaluated at the projected mean. The screen-space covariance is

```math
\Sigma' = J W \Sigma W^\top J^\top
```

$J$ is a first-order Taylor approximation that linearizes the perspective transform locally at the mean. The contribution of Gaussian $i$ at screen position $\mathbf{p}$ is then

```math
\alpha_i \exp\!\left(-\tfrac{1}{2}\,\Delta\mathbf{p}^\top \Sigma'^{-1} \Delta\mathbf{p}\right), \qquad \Delta\mathbf{p} = \mathbf{p} - \pi(\mu_i)
```

where $\pi$ denotes perspective projection and $\alpha_i = \text{sigmoid}(\sigma_i)$.

**View-dependent color.** Degree-3 spherical harmonics provide 16 basis functions per channel. The prepass evaluates them for the current view direction and writes a single RGB value per Gaussian.

### Depth Sorting and Prepass Caching

Alpha compositing requires Gaussians in back-to-front order, which changes with every camera rotation.

**Radix sort.** An LSD radix sort reorders the Gaussian index array each rotation. Four passes — one per byte of the sort key, LSB to MSB — each consist of a histogram, a prefix scan, and a stable scatter: 12 compute dispatches total versus 190+ for a bitonic sort. The sort key is the bitwise complement of the NDC depth, which maps positive floats to descending integer order. Stability within each workgroup is achieved without atomics: one thread per bucket scans the workgroup's 256 elements in thread-index order, assigning ranks to disjoint positions in shared memory.

**Prepass caching.** Without caching, SH evaluation and covariance projection would execute four times per Gaussian per frame — once per quad vertex. A compute prepass writes a 12-float record per Gaussian (NDC position, screen-space covariance inverse, RGB, opacity) that all four vertices read identically. On static frames neither the prepass nor the sort fires.