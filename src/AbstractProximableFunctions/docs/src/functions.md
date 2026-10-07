# Main functions

## General proximal/projection functionalities

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

## Iterative solver utilities

```@docs
exact_argmin
```

```@docs
FISTA_options
```

## Implemented proximable functions

```@docs
LinearAlgebra.norm(T::DataType, N::Number, P::Number)
```

```@docs
mixed_norm(T::DataType, D::Number, N1::Number, N2::Number)
```