# RetrospectiveMotionCorrectionMRI

Retrospective correction of rigid-body motion in 3D Cartesian MRI. The motion parameters (per k-space line, or per interpolation node) and the image are estimated jointly, by alternating

- **image reconstruction**: FISTA with a constraint on the structure-guided total variation, where the edges of a reference scan (e.g. with a different contrast) guide the regularization;
- **motion-parameter estimation**: Gauss-Newton on the rigid-motion parameters, using a NUFFT forward model with per-line rotations and translations.

See the [documentation](https://oscarvanderheide.github.io/RetrospectiveMotionCorrectionMRI/) for the theory and a step-by-step example.

## Installation

```julia
pkg> add https://github.com/oscarvanderheide/RetrospectiveMotionCorrectionMRI
```

The package is self-contained: the formerly separate packages `AbstractLinearOperators`, `AbstractProximableFunctions`, `FastSolversForWeightedTV` and `UtilitiesForMRI` are included as submodules (in `src/`) and re-exported.

## Usage

- `examples/example_basic_usage.jl`: simulated example.
- `examples/generic_correction.jl`: template for correcting a motion-corrupted scan (`.mat`) using a reference scan.

### Performance options

| Option | Where | Effect |
|---|---|---|
| `tol` | `nfft_linop(X, K; tol=1f-4)` | NUFFT accuracy. `1f-4` is ~3x faster than the default `1f-6` and changes results negligibly. |
| `warmstart` | `gradient_norm(...; warmstart=true)` | Warm start of the (inexact) projection onto the TV constraint. More accurate for the same number of inner iterations, but results depend on the history of previous calls. Default `false`. |
| `toeplitz` | `image_reconstruction_options(...; toeplitz=true)` | Evaluates `F'F` with FFTs (Toeplitz embedding) instead of NUFFTs. Pays off when many iterations are run per reconstruction. Default `false`. |

CPU code is multithreaded: start Julia with e.g. `julia -t 8`.

### GPU

GPU support (NVIDIA, CUDA) is provided through a package extension that is loaded when [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and [NonuniformFFTs.jl](https://github.com/jipolanco/NonuniformFFTs.jl) are loaded. Move the NUFFT operator, data, reference image and initial image to the GPU; motion parameters stay on the CPU:

```julia
using CUDA, NonuniformFFTs, RetrospectiveMotionCorrectionMRI

F = cu(nfft_linop(X, K; tol=1f-4))
d = cu(d); reference = cu(reference); u0 = cu(u0)
# ... set up options as usual (build the regularization from the GPU reference)
u, θ = motion_corrected_reconstruction(F, d, u0, θ0, options)
u = Array(u)
```

In `examples/generic_correction.jl`, set `use_gpu = true`.

## Tests

```julia
pkg> test RetrospectiveMotionCorrectionMRI
```

This runs the tests of the main package and of the submodules. GPU tests run only when a functional CUDA GPU is available.

## Credits

Originally developed by Gabrio Rizzuti ([grizzuti/RetrospectiveMotionCorrectionMRI](https://github.com/grizzuti/RetrospectiveMotionCorrectionMRI) and the separate packages listed above). Updated for recent Julia versions by Oscar van der Heide, restructured as a single package and applied to clinical data by Pablo Domínguez, and optimized (multithreaded CPU and GPU) by Oscar van der Heide.
