# Main functions

```@docs
gradient_norm
```

## Important utilities from ancillary packages

We list the most important utilities from `AbstractProximableFunctions` that are meant to be routinely combined with `FastSolversForWeightedTV`, such as tools to define iterative solver options, and functions to call the proximal/projection operators.

```@docs
prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
```

```@docs
prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}}
```

```@docs
proj(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
```

```@docs
proj(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}}
```

```@docs
exact_argmin
```

```@docs
FISTA_options
```