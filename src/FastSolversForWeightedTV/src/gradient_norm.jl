#: Utilities for norm functions

export gradient_norm, reset_warmstart!


# Gradient mixed norm

"""
    gradient_norm(p1, p2, n, h; weight=nothing,
                                complex=false,
                                options=exact_argmin(),
                                warmstart=false)

Returns the regularization function associated to the weighted gradient mixed norm ``g(\\mathbf{u})=||\\mathrm{A}\\nabla\\mathbf{u}||_{p1,p2}``.

The mixed norm is specified by `p1` and `p2`. For `p1=2`, `p2=1`, and `weight=nothing`, for example, one gets conventional TV. Other supported options are `p2=2` or `p2=Inf`.

The Cartesian grid geometry is determined by the grid size `n` and spacing `h`. For instance, in 3D, `n=(64, 128, 256)`, `h=(1f0, 2f0, 3f0)`.

The linear operator ``A`` is specified via the keyword `weight`. Note that this weight must be initialized via the tools contained in the package `AbstractLinearOperators` (see Section [Getting started](@ref) for an example).

Complex or real inputs are handled via the keyword `complex`. Set `complex=true` for complex image input.

For the evaluation of the associated proximal operator, one must specify a solver with the keyword `options`. Dedicated solvers are offered by the package `AbstractProximableFunctions.jl` (e.g. FISTA, see Section [Getting started](@ref) for some basic usage options).

With `warmstart=true`, the (iterative) projection on the sublevel set ``\\{g\\le\\varepsilon\\}`` is initialized with the dual solution of the previous projection, rather than with zero. Since the inner FISTA solver is typically run for a small fixed number of iterations, this generally yields more accurate projections (allowing fewer inner iterations), but it also makes the result depend on the history of previous calls. Use [`reset_warmstart!`](@ref) to discard the stored solution. Only supported for 3D CPU arrays with `weight=nothing` or a `ProjVectorField` weight (ignored otherwise).
"""
function gradient_norm(P1::Number, P2::Number, n::NTuple{N,Int64}, h::NTuple{N,T}; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false, options::AbstractArgminOptions=exact_argmin(), warmstart::Bool=false) where {T<:Real,N}
    complex ? (CT = Complex{T}) : (CT = T)
    if isnothing(weight) || weight isa ProjVectorField{CT,N+1}
        A∇ = weighted_gradient_operator(CT, n, h; weight=weight, warmstart=warmstart) # fused evaluation
    else
        warmstart && @warn "warmstart is only supported for weight=nothing or a ProjVectorField weight, and will be ignored"
        ∇ = gradient_operator(n, h; complex=complex)
        A∇ = weight*∇
    end
    return weighted_prox(mixed_norm(CT,N,P1,P2; pareto_tol=pareto_tol), A∇; options=options)
end


"""
    reset_warmstart!(g)

Discards the dual solution stored by a regularization function created with `gradient_norm(...; warmstart=true)`, so that the next projection starts from zero.
"""
function reset_warmstart!(g::WeightedProximableFunction)
    g.linear_operator isa WeightedGradientOperator && (g.linear_operator.dual[] = nothing)
    return g
end
