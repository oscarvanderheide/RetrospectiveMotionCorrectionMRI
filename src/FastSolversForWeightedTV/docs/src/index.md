# Introduction

## Reference-guided TV

FastSolversForWeightedTV.jl allows the computation of the proximal operator or projection operator associated to the reference-guided total variation regularization (TV) introduced in `[1]` (see also `[2]` for an application to motion correction in MRI). This implementation comprises both 2D and 3D, and supports GPU acceleration.

The definition of the reference-guided TV regularization is:
```math
\mathrm{TV}(\mathbf{u}; \mathbf{v})=||\Pi(\mathbf{v})\nabla\mathbf{u}||_{2,1}=\sum_i ||\Pi(\mathbf{v})|_i\nabla\mathbf{u}|_i||_2,\qquad\Pi(\mathbf{v})|_i=\mathrm{I}_{d\times d}-\hat{\nabla\mathbf{v}}|_i{}^{\mathrm{H}}\hat{\nabla\mathbf{v}}|_i.
```
Here, the reference image (2D/3D) is denoted by ``\mathbf{v}``, ``\nabla`` is the discretized gradient operator, and the index ``i`` refers to the related grid point. ``\mathrm{I}_{d\times d}`` is the ``d\times d`` identity matrix (where ``d=2,3``, depending on the dimension of the problem). The normalized reference gradient field is defined by:
```math
\hat{\nabla\mathbf{v}}|_i=\dfrac{\nabla\mathbf{v}|_i}{\sqrt{||\nabla\mathbf{v}|_i||_2^2+\eta^2}},
```
where the constant ``\eta>0`` stabilizes the division.

## Proximal and projection operators

The main functionalities provided by this package are related to the computation of the proximal and projection operators. For a generic convex functional ``g``, the proximal operator is defined as
```math
\mathrm{prox}_{\lambda,g}(\mathbf{w})=\arg\min_{\mathbf{u}}\dfrac{1}{2}||\mathbf{u}-\mathbf{w}||^2+\lambda{}g(\mathbf{u}).
```
Similarly, the projection operator is defined as
```math
\mathrm{proj}_{\varepsilon, g}(\mathbf{w})=\arg\min_{g(\mathbf{u})\le\varepsilon}\dfrac{1}{2}||\mathbf{u}-\mathbf{w}||^2.
```

# Related publications

1. Ehrhardt, M. J., and Betcke, M. M., (2015). Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation (https://arxiv.org/abs/1511.06631), _SIAM J. IMAGING SCIENCES_, **9(3)**, 1084-1106, doi:[10.1137/15M1047325](https://doi.org/10.1137/15M1047325)
2. Rizzuti, G., Sbrizzi, A., and van Leeuwen, T., (2022). Joint Retrospective Motion Correction and Reconstruction for Brain MRI With a Reference Contrast, _IEEE Transaction on Computational Imaging_, **8**, 490-504, doi:[10.1109/TCI.2022.3183383](hhtps://doi.org/10.1109/TCI.2022.3183383)