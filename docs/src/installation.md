# [Installation instructions](@id install)

In the Julia REPL, type `]` and

```julia
pkg> add https://github.com/oscarvanderheide/RetrospectiveMotionCorrectionMRI
```

The package is self-contained: `AbstractLinearOperators`, `AbstractProximableFunctions`, `FastSolversForWeightedTV` and `UtilitiesForMRI`, which used to be separate (unregistered) packages, are included as submodules and re-exported by `RetrospectiveMotionCorrectionMRI`.

For GPU support, also install [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and [NonuniformFFTs.jl](https://github.com/jipolanco/NonuniformFFTs.jl) (see the README).
