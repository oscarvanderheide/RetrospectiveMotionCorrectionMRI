#: Miscellanea

export spectral_radius
export test_grad
export set_options


function spectral_radius(A::Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}; x::Union{Nothing,AbstractArray{T,N}}=nothing, niter::Int64=10) where {T,N}
    if isnothing(x)
        A isa AbstractMatrix && (x = randn(T, size(A,2)))
        A isa AbstractLinearOperator && (x = randn(T, domain_size(A)))
    end
    x = x/norm(x)
    y = similar(x)
    ρ = real(T)(0)
    for _ = 1:niter
        y .= A*x
        ρ = norm(y)/norm(x)
        x .= y/norm(y)
    end
    return ρ
end

function test_grad(fun::AbstractDifferentiableFunction{CT,N}, x::AbstractArray{CT,N}; step::T=T(1e-4), rtol::T=eps(T)) where {T<:Real,N,CT<:Union{T,Complex{T}}}

    dx = convert(typeof(x), randn(CT, size(x))); dx *= norm(x)/norm(dx)
    Δx = gradeval(fun, x)
    fp1 = funeval(fun, x+T(0.5)*step*dx)
    fm1 = funeval(fun, x-T(0.5)*step*dx)
    return isapprox((fp1-fm1)/step, real(dot(dx, Δx)); rtol=rtol)

end

set_options(op::DiffPlusProxFunction, options::AbstractArgminOptions) = +(op.diff, op.prox; options=options)
set_options(op::ProxPlusIndicator, options::AbstractArgminOptions) = +(op.prox, op.indicator; options=options)
set_options(op::WeightedProximableFunction, options::AbstractArgminOptions) = weighted_prox(op.prox, op.linear_operator; options=options)
set_options(op::WeightedProxPlusIndicator, options::AbstractArgminOptions) = +(op.wprox, op.indicator; options=options)