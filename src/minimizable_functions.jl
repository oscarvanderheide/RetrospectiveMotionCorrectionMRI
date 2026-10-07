#: Optimization utilities

export ExactArgmin, exact_argmin
export ArgminFISTA, FISTA_options, set_Lipschitz_constant, fun_history


## Fallback behavior for argmin

not_implemented() = error("This method has not been implemented for this function!")

argmin!(::AbstractMinimizableFunction{T,N}, ::AT, ::AbstractArgminOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()
prox!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractArgminOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractArgminOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::AbstractProjectionableSet{T,N}, ::AbstractArgminOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()


## Exact options

struct ExactArgmin<:AbstractArgminOptions end

"""
    exact_argmin()

Returns exact optimization options for the computation of proximal/projection operators. It might results in an error if the regularization functional does not implement analytically-defined proximal/projection operators (e.g. TV!).
"""
exact_argmin() = ExactArgmin()


# Differentiable+proximable functions

struct DiffPlusProxFunction{T,N}<:AbstractMinimizableFunction{T,N}
    diff::AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    options::AbstractArgminOptions
end

Base.:+(f::AbstractDifferentiableFunction{T,N}, g::AbstractProximableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = DiffPlusProxFunction{T,N}(f, g, options)
Base.:+(g::AbstractProximableFunction{T,N}, f::AbstractDifferentiableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = +(f, g; options=options)


## FISTA options

struct ArgminFISTA<:AbstractArgminOptions
    Lipschitz_constant::Union{Nothing,Real}
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{<:Real}}
end

"""
    FISTA_options(L; Nesterov=true,
                     reset_counter=nothing,
                     niter=nothing,
                     verbose=false,
                     fun_history=false)

Returns FISTA iterative solver options for the general optimization problem:

``\\min_{\\mathbf{x}}f(\\mathbf{x})+g(\\mathbf{x})``

where ``g`` is a "proximable" function. It can be used to define options for the proximal or projection operators.

The parameter ``L`` ideally should be chosen as ``L\\ge\\mathrm{Lip}\\nabla f`` (=Lipschitz constant of the gradient) and is problem specific.

Nesterov acceleration is set by `Nesterov=true`, while `reset_counter` is the number of iteration after which the Nesterov momentum is reset. The total number of iterations is determined by `niter`.

For debugging, set `verbose=true` and/or `fun_history=true` (the latter allows storing the history of ``f(\\mathbf{x})``, which can be retrieved by `fun_history(options)` after minimization).

## Important note

When setting the options for proximal or projection operators, follow the recommendations for each specific proximal function on how to choose ``L``. The underlying optimization may be based on algebraic reformulations of the optimization problem ``\\min_{\\mathbf{x}}1/2||\\mathbf{x}-\\mathbf{y}||^2+\\lambda{}g(\\mathbf{x})``, so the assumption that ``f(\\mathbf{x})=1/2||\\mathbf{x}-\\mathbf{y}||^2`` is not generally correct. This note is relevant when computing the proximal operator of `WeightedProximalFunction`'s for which the problem ``\\min_{\\mathbf{x}}1/2||\\mathbf{x}-\\mathbf{y}||^2+\\lambda{}g(A\\mathbf{x})`` is transformed to ``\\min_{\\mathbf{p}}1/2||\\lambda A^*\\mathbf{p}-\\mathbf{y}||^2+\\lambda g^*(\\mathbf{p})``. The Lipschitz constant in this case should be ``\\lambda^2\\rho(A)``. For convenience of use, however, the Lipschitz constant is expected to be just ``\\rho(A)``.
"""
function FISTA_options(L::Union{Nothing,Real};
                       Nesterov::Bool=true,
                       reset_counter::Union{Nothing,Integer}=nothing,
                       niter::Union{Nothing,Integer}=nothing,
                       verbose::Bool=false,
                       fun_history::Bool=false)
    isnothing(L) ? (T = Real) : (T = typeof(L))
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    return ArgminFISTA(L, Nesterov, reset_counter, niter, verbose, fval)
end

Lipschitz_constant(options::ArgminFISTA) = options.Lipschitz_constant
set_Lipschitz_constant(options::ArgminFISTA, L::Real) = ArgminFISTA(L, options.Nesterov, options.reset_counter, options.niter, options.verbose, options.fun_history)
fun_history(options::ArgminFISTA) = options.fun_history

argmin!(fun::DiffPlusProxFunction{CT,N}, initial_estimate::AT, options::ArgminFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = (x .= initial_estimate; argmin!(fun, x, options))

function argmin!(fun::DiffPlusProxFunction{CT,N}, x::AT, options::ArgminFISTA) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}
    # - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

    # Initialization
    options.Nesterov ? (x_ = similar(x)) : (x_ = x)
    xprev_ = deepcopy(x)
    gradient = similar(x)
    L = T(Lipschitz_constant(options))
    isnothing(options.reset_counter) ? (counter = nothing) : (counter = 0)
    t0 = T(1)
    diff_fun = fun.diff
    prox_fun = fun.prox

    # Optimization loop
    @inbounds for n = 1:options.niter

        # Compute gradient
        if options.verbose || ~isnothing(fun_history(options))
            fval_n = fungradeval!(diff_fun, x, gradient)
        else
            gradeval!(diff_fun, x, gradient)
        end
        ~isnothing(fun_history(options)) && (fun_history(options)[n] = fval_n)

        # Print current iteration
        options.verbose && (@info string("Iter: ", n, ", fval: ", fval_n))

        # Update
        prox!(x-gradient/L, 1/L, prox_fun, x_)

        # Nesterov acceleration
        if options.Nesterov
            t = (1+sqrt(1+4*t0^2))/2
            (n == 1) ? (x .= x_) : (x .= x_+(t0-1)/t*(x_-xprev_))
            ~isnothing(counter) && (counter += 1)

            # Update
            t0 = t
            xprev_ .= x_

            # Reset momentum
            ~isnothing(options.reset_counter) && (counter >= options.reset_counter) && (t0 = T(1))
        end

    end

    return x

end