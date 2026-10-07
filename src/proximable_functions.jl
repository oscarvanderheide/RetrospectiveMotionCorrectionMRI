#: Proximable functions

export ZeroProx, zero_prox
export IndicatorFunction, indicator
export WeightedProximableFunction, weighted_prox
export conjugate


## Zero proximable function

struct ZeroProx{T,N}<:AbstractProximableFunction{T,N} end

zero_prox(T::DataType, N::Number) = ZeroProx{T,N}()

funeval(::ZeroProx{CT,N}, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = T(0)
prox!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, ::ExactArgmin, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)
proj!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, ::ExactArgmin, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)


## Indicator functions

struct IndicatorFunction{T,N}<:AbstractProximableFunction{T,N}
    set::AbstractProjectionableSet{T,N}
end

indicator(C::AbstractProjectionableSet) = IndicatorFunction(C)

funeval(δC::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ δC.set) ? T(0) : T(Inf)

prox!(y::AT, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgminOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proj!(y, δ.set, options, x)
proj!(y::AT, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgminOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proj!(y, δ.set, options, x)

options(δ::IndicatorFunction) = options(δ.set)


## Weighted proximable functions

struct WeightedProximableFunction{T,N,M}<:AbstractProximableFunction{T,N}
    prox::AbstractProximableFunction{T,M}
    linear_operator::AbstractLinearOperator{T,N,M}
    options::AbstractArgminOptions
end

weighted_prox(g::AbstractProximableFunction{T,M}, A::AbstractLinearOperator{T,N,M}; options::AbstractArgminOptions=exact_argmin()) where {T,N,M} = WeightedProximableFunction{T,N,M}(g, A, options)
Base.:∘(g::AbstractProximableFunction{T,M}, A::AbstractLinearOperator{T,N,M}; options::AbstractArgminOptions=exact_argmin()) where {T,N,M} = weighted_prox(g, A; options=options)
options(g::WeightedProximableFunction) = g.options

funeval(g::WeightedProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = g.prox(g.linear_operator*x)

function prox!(y::AT, λ::T, g::WeightedProximableFunction{CT,N}, options::ArgminFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*g.linear_operator', y)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    options = set_Lipschitz_constant(options, λ^2*Lipschitz_constant(options))
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return x .= y-λ*(g.linear_operator'*p)

end

function proj!(y::AT, ε::T, g::WeightedProximableFunction{CT,N}, options::ArgminFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}

    # Objective function (dual problem)
    f = leastsquares_misfit(g.linear_operator', y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return x .= y-g.linear_operator'*p

end


## Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:AbstractProximableFunction{T,N}
    fun::AbstractProximableFunction{T,N}
end

conjugate(g::AbstractProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction) = g.fun

function prox!(y::AbstractArray{CT,N}, λ::T, g::ConjugateProximableFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    prox!(y/λ, 1/λ, g.fun, options, x)
    return x .= y-λ*x
end

options(g::ConjugateProximableFunction) = options(g.fun)


## Linear algebra utilities

struct ScaledProximableFunction{T,N}<:AbstractProximableFunction{T,N}
    scale::Real
    fun::AbstractProximableFunction{T,N}
end

Base.:*(scale::T, fun::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFunction{CT,N}(scale, fun)
Base.:*(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

funeval(fun::ScaledProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)

prox!(y::AbstractArray{CT,N}, λ::T, g::ScaledProximableFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = prox!(y, λ*g.scale, g.fun, options, x)
proj!(y::AbstractArray{CT,N}, ε::T, g::ScaledProximableFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proj!(y, g.scale/ε, g.fun, options, x)

options(g::ScaledProximableFunction) = options(g.fun)


## Proximable + indicator

struct ProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    indicator::IndicatorFunction{T,N}
    options::AbstractArgminOptions
end

Base.:+(g::AbstractProximableFunction{T,N}, δ::IndicatorFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = ProxPlusIndicator{T,N}(g, δ, options)
Base.:+(δ::IndicatorFunction{T,N}, g::AbstractProximableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = +(g, δ; options=options)

funeval(g::ProxPlusIndicator{T,N}, x::AbstractArray{T,N}) where {T,N} = g.prox(x)+g.indicator(x)

struct _ProxPlusIndicatorProxObj{T,N}<:AbstractDifferentiableFunction{T,N}
    y::AbstractArray{T,N}
    λ::Real
    set::AbstractProjectionableSet{T,N}
end

_prox_plus_indicator_proxobj(y::AbstractArray{CT,N}, λ::T, C::AbstractProjectionableSet{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = _ProxPlusIndicatorProxObj{CT,N}(y, λ, C)

function funeval(f::_ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = proj(r, f.set)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2
end

function gradeval!(f::_ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = proj(r, f.set)
    return g .= -f.λ*xC
end

function fungradeval!(f::_ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = proj(r, f.set)
    g .= -f.λ*xC
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2, g
end

function prox!(y::AbstractArray{CT,N}, λ::T, g::ProxPlusIndicator{CT,N}, options::ArgminFISTA, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = _prox_plus_indicator_proxobj(y, λ, g.indicator.set)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    options = set_Lipschitz_constant(options, λ^2*Lipschitz_constant(options))
    p0 = similar(y); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return proj!(y-λ*p, g.indicator.set, x)

end

function proj!(y::AbstractArray{CT,N}, ε::T, g::ProxPlusIndicator{CT,N}, options::ArgminFISTA, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = _prox_plus_indicator_proxobj(y, T(1), g.indicator.set)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return proj!(y-p, g.indicator.set, x)

end

options(g::ProxPlusIndicator) = g.options


## Weighted proximable + indicator

struct WeightedProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N}
    wprox::WeightedProximableFunction{T,N}
    indicator::IndicatorFunction{T,N}
    options::AbstractArgminOptions
end

Base.:+(g::WeightedProximableFunction{T,N}, δ::IndicatorFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = WeightedProxPlusIndicator{T,N}(g, δ, options)
Base.:+(δ::IndicatorFunction{T,N}, g::WeightedProximableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = +(g, δ; options=options)

funeval(g::WeightedProxPlusIndicator{T,N}, x::AbstractArray{T,N}) where {T,N} = g.wprox(x)+g.indicator(x)

struct _WeightedProxPlusIndicatorProxObj{T,N1,N2}<:AbstractDifferentiableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N1}
    λ::Real
    set::AbstractProjectionableSet{T,N1}
end

_wprox_plus_indicator_proxobj(A::AbstractLinearOperator{CT,N1,N2}, y::AbstractArray{CT,N1}, λ::T, C::AbstractProjectionableSet{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = _WeightedProxPlusIndicatorProxObj{CT,N1,N2}(A, y, λ, C)

function funeval(f::_WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = proj(r, f.set)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2
end

function gradeval!(f::_WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}, g::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = proj(r, f.set)
    return g .= -f.λ*(f.linear_operator*xC)
end

function fungradeval!(f::_WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}, g::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = proj(r, f.set)
    g .= -f.λ*(f.linear_operator*xC)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2, g
end

function prox!(y::AbstractArray{CT,N}, λ::T, g::WeightedProxPlusIndicator{CT,N}, options::ArgminFISTA, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    wprox = g.wprox
    f = _wprox_plus_indicator_proxobj(wprox.linear_operator, y, λ, g.indicator.set)+λ*conjugate(wprox.prox)

    # Minimization (dual variable)
    options = set_Lipschitz_constant(options, λ^2*Lipschitz_constant(options))
    p0 = similar(y, range_size(wprox.linear_operator)); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return proj!(y-λ*(wprox.linear_operator'*p), g.indicator.set, x)

end

function proj!(y::AbstractArray{CT,N}, ε::T, g::WeightedProxPlusIndicator{CT,N}, options::ArgminFISTA, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    wprox = g.wprox
    f = _wprox_plus_indicator_proxobj(wprox.linear_operator, y, T(1), g.indicator.set)+conjugate(indicator(wprox.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(wprox.linear_operator)); p0 .= 0
    p = argmin(f, p0, options)

    # Dual to primal solution
    return proj!(y-wprox.linear_operator'*p, g.indicator.set, x)

end

options(g::WeightedProxPlusIndicator) = g.options