#: Main abstract functional types

export AbstractEvaluableFunction, funeval
export AbstractMinimizableFunction, AbstractArgminOptions, argmin, argmin!, options
export AbstractProximableFunction, prox, prox!, proj, proj!
export AbstractProjectionableSet
export AbstractDifferentiableFunction, gradeval, gradeval!, fungradeval, fungradeval!


## Evaluable functions

abstract type AbstractEvaluableFunction{T,N} end
# funeval(fun::AbstractEvaluableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

(fun::AbstractEvaluableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = funeval(fun, x)


## Minimizable functions

abstract type AbstractMinimizableFunction{T,N}<:AbstractEvaluableFunction{T,N} end
abstract type AbstractArgminOptions end
# argmin!(fun::AbstractMinimizableFunction{T,N}, initial_estimate::AT, options::AbstractArgminOptions, x::AT) where {T<:RealOrComplex,N,AT<:AbstractArray{T,N}} = ...
# options(fun::AbstractMinimizableFunction{T,N}) where {T,N} = ...

Base.argmin(fun::AbstractMinimizableFunction{T,N}, initial_estimate::AbstractArray{T,N}, options::AbstractArgminOptions) where {T<:RealOrComplex,N} = argmin!(fun, initial_estimate, options, similar(initial_estimate))
Base.argmin(fun::AbstractMinimizableFunction{T,N}, initial_estimate::AbstractArray{T,N}) where {T<:RealOrComplex,N} = argmin!(fun, initial_estimate, options(fun), similar(initial_estimate))
argmin!(fun::AbstractMinimizableFunction{T,N}, initial_estimate::AT, x::AT) where {T<:RealOrComplex,N,AT<:AbstractArray{T,N}} = argmin!(fun, initial_estimate, options(fun), x)

options(f::AbstractMinimizableFunction) = exact(f)


## Proximable functions

abstract type AbstractProximableFunction{T,N}<:AbstractEvaluableFunction{T,N} end
# prox!(g::AbstractProximableFunction{CT,N}, y::AT, λ::T, options::AbstractArgminOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = ...
# proj!(g::AbstractProximableFunction{CT,N}, y::AT, ε::T, options::AbstractArgminOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = ...
# options(g::AbstractProximableFunction) = ...

"""
    prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

Computes the proximal operator ``\\mathrm{prox}_{\\lambda,g}(\\mathbf{y})``.
"""
prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = prox!(y, λ, g, options(g), similar(y))
"""
    prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}}

Computes the proximal operator with specified options.

See [`prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}`](@ref)
"""
prox(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}} = prox!(y, λ, g, options, similar(y))
prox!(y::AT, λ::T, g::AbstractProximableFunction{CT,N}, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = prox!(y, λ, g, options(g), x)

"""
    proj(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

Computes the projection operator ``\\mathrm{proj}_{\\lambda,g}(\\mathbf{y})``.
"""
proj(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proj!(y, ε, g, options(g), similar(y))
"""
    proj(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}}

Computes the projection operator with specified options.

See [`proj(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}`](@ref)
"""
proj(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminOptions) where {T<:Real,N,CT<:RealOrComplex{T}} = proj!(y, ε, g, options, similar(y))
proj!(y::AT, ε::T, g::AbstractProximableFunction{CT,N}, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proj!(y, ε, g, options(g), x)

options(::AbstractProximableFunction) = exact_argmin()


## Projection sets

abstract type AbstractProjectionableSet{T,N} end
# Base.in(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = ...
# proj!(x::AT, C::AbstractProjectionableSet{T,N}, options::AbstractArgminOptions, y::AT) where {T,N,AT<:AbstractArray{T,N}} = ...

proj(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = proj!(x, C, options(C), similar(x))
proj(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractArgminOptions) where {T,N} = proj!(x, C, options, similar(x))
proj!(x::AT, C::AbstractProjectionableSet{T,N}, y::AT) where {T,N,AT<:AbstractArray{T,N}} = proj!(x, C, options(C), y)

options(::AbstractProjectionableSet) = exact_argmin()


## Differentiable functions

abstract type AbstractDifferentiableFunction{T,N}<:AbstractEvaluableFunction{T,N} end
# gradeval!(fun::AbstractDifferentiableFunction{T,N}, x::AT, gradient::AT) where {T,N,AT<:AbstractArray{T,N}} = ...
# fungradeval!(fun::AbstractDifferentiableFunction{T,N}, x::AT, gradient::AT) where {T,N,AT<:AbstractArray{T,N}} = ...

gradeval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); gradeval!(fun, x, g); return g)
fungradeval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); fval = fungradeval!(fun, x, g); return (fval, g))