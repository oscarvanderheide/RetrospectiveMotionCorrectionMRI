# Main functions


## Rigid motion parameter estimation

```@docs
parameter_estimation(F::RetrospectiveMotionCorrectionMRI.StructuredNFFTtype2LinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{T}, options::RetrospectiveMotionCorrectionMRI.ParameterEstimationOptionsDiff) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
parameter_estimation_options(; niter::Integer=10,
                                        steplength::Real=1f0,
                                        λ::Real=0f0,
                                        scaling_diagonal::Real=0f0, scaling_mean::Real=0f0, scaling_id::Real=0f0,
                                        reg_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        interp_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        calibration::Bool=false,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
```

## Rigid registration utilities

```@docs
rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, options::RetrospectiveMotionCorrectionMRI.ParameterEstimationOptionsDiff; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing, nscales::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}
```

```@docs
rigid_registration_options(; niter::Integer=10, verbose::Bool=false, fun_history::Bool=false)
```