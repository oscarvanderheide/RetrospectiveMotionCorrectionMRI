#: Differentiable functions

export ProxObjFun, ProjObjFun, prox_objfun, proj_objfun
export LeastSquaresMisfit, leastsquares_misfit, leastsquares_solve, leastsquares_solve!


## Differentiable function linear algebra

struct ScaledDiffFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    scale::Real
    fun::AbstractDifferentiableFunction{T,N}
end

Base.:*(scale::T, fun::AbstractDifferentiableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledDiffFunction{CT,N}(scale, fun)
Base.:*(fun::AbstractDifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::AbstractDifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

funeval(fun::ScaledDiffFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)
gradeval!(fun::ScaledDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}} = (gradeval!(fun.fun, x, g); g .*= fun.scale; return g)
fungradeval!(fun::ScaledDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}} = (fval = fungradeval!(fun.fun, x, g); g .*= fun.scale; return fun.scale*fval)

struct PlusDiffFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    fun1::AbstractDifferentiableFunction{T,N}
    fun2::AbstractDifferentiableFunction{T,N}
end

Base.:+(fun1::AbstractDifferentiableFunction{T,N}, fun2::AbstractDifferentiableFunction{T,N}) where {T,N} = PlusDiffFunction{T,N}(fun1, fun2)

funeval(fun::PlusDiffFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)+fun.fun2(x)
gradeval!(fun::PlusDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}} = (gradeval!(fun.fun1, x, g); g .+= gradeval(fun.fun2, x); return g)
function fungradeval!(fun::PlusDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}}
    fval = fungradeval!(fun.fun1, x, g)
    fval2, g2 = fungradeval(fun.fun2, x)
    g .+= g2
    return fval+fval2
end

struct MinusDiffFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    fun1::AbstractDifferentiableFunction{T,N}
    fun2::AbstractDifferentiableFunction{T,N}
end

Base.:-(fun1::AbstractDifferentiableFunction{T,N}, fun2::AbstractDifferentiableFunction{T,N}) where {T,N} = MinusDiffFunction{T,N}(fun1, fun2)

funeval(fun::MinusDiffFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)-fun.fun2(x)
gradeval!(fun::MinusDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}} = (gradeval!(fun.fun1, x, g); g .-= gradeval(fun.fun2, x); return g)
function fungradeval!(fun::MinusDiffFunction{T,N}, x::AT, g::AT) where {T,N,AT<:AbstractArray{T,N}}
    fval = fungradeval!(fun.fun1, x, g)
    fval2, g2 = fungradeval(fun.fun2, x)
    g .-= g2
    return fval-fval2
end


## Proxy objective function

struct ProxObjFun{T,N}<:AbstractDifferentiableFunction{T,N}
    fun::AbstractProximableFunction{T,N}
    weight::Real
    options::AbstractArgminOptions
end

prox_objfun(g::AbstractProximableFunction{CT,N}, λ::T; options::AbstractArgminOptions=options(g)) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxObjFun{CT,N}(g, λ, options)

function funeval(fun::ProxObjFun{T,N}, y::AbstractArray{T,N}) where {T,N}
    x̄ = prox(y, fun.weight, fun.fun, fun.options)
    return norm(x̄-y)^2/2+fun.weight*fun.fun(x̄)
end

function gradeval!(fun::ProxObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-prox(y, fun.weight, fun.fun, fun.options)
    return g
end

function fungradeval!(fun::ProxObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    x̄ = prox(y, fun.weight, fun.fun, fun.options)
    g .= y-x̄
    return norm(g)^2/2+fun.weight*fun.fun(x̄)
end


## Projection objective function

struct ProjObjFun{T,N}<:AbstractDifferentiableFunction{T,N}
    fun::AbstractProximableFunction{T,N}
    level::Real
    options::AbstractArgminOptions
end

proj_objfun(g::AbstractProximableFunction{CT,N}, ε::T; options::AbstractArgminOptions=options(g)) where {T<:Real,N,CT<:RealOrComplex{T}} = ProjObjFun{CT,N}(g, ε, options)

funeval(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}) where {T,N} = norm(proj(y, fun.level, fun.fun, fun.options)-y)^2/2

function gradeval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-proj(y, fun.level, fun.fun, fun.options)
    return g
end

function fungradeval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-proj(y, fun.weight, fun.fun, fun.options)
    return norm(g)^2/2
end


## Least-squares misfit

struct LeastSquaresMisfit{T,N1,N2}<:AbstractDifferentiableFunction{T,N1}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    known_term::AbstractArray{T,N2}
end

leastsquares_misfit(A::AbstractLinearOperator{T,N1,N2}, y::AbstractArray{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)

funeval(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2} = norm(f.linear_operator*x-f.known_term)^2/2

function gradeval!(f::LeastSquaresMisfit{T,N1,N2}, x::AT, gradient::AT) where {T<:RealOrComplex,N1,N2,AT<:AbstractArray{T,N1}}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return gradient
end

function fungradeval!(f::LeastSquaresMisfit{T,N1,N2}, x::AT, gradient::AT) where {T<:RealOrComplex,N1,N2,AT<:AbstractArray{T,N1}}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return norm(f.linear_operator*x-f.known_term)^2/2
end


## Least-squares linear problem routines

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::AbstractArgminOptions, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = argmin!(leastsquares_misfit(A, b)+g, initial_estimate, options, x)

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::AbstractArgminOptions) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = leastsquares_solve!(A, b, g, initial_estimate, options, similar(initial_estimate))