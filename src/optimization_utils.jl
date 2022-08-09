# Optimization utilities:
# - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

export OptionsFISTA, OptimiserFISTA
export reset!, linesearch_backtracking, spectral_radius


## FISTA

struct OptionsFISTA{T}
    Lipschitz_constant::T
    prox::Function
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
end

FISTA_options(L::T, prox::Function; Nesterov::Bool=true, reset_counter::Union{Nothing,Integer}=nothing) where {T<:Real} = OptionsFISTA{T}(L, prox, Nesterov, reset_counter)

mutable struct OptimiserFISTA{T}<:AbstractOptimiserFISTA{T}
    options::OptionsFISTA{T}
    t::T
    counter::Union{Nothing,Integer}
end

OptimiserFISTA(L::T, prox::Function; t::T=T(1), Nesterov::Bool=true, reset_counter::Union{Nothing,Integer}=nothing) where {T<:Real} = OptimiserFISTA{T}(FISTA_options(L, prox; Nesterov=Nesterov, reset_counter=reset_counter), t, isnothing(reset_counter) ? nothing : 0)

function reset!(opt::AbstractOptimiserFISTA{T}) where {T<:Real}
    opt.t = T(1)
    ~isnothing(opt.counter) && (opt.counter = 0)
end

function Flux.Optimise.apply!(opt::AbstractOptimiserFISTA{T}, x::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Simplifying notation
    L = opt.options.Lipschitz_constant
    prox = opt.options.prox
    Nesterov = opt.options.Nesterov
    t0 = opt.t
    reset_counter = opt.options.reset_counter

    # Gradient + proxy update
    steplength = T(1)/L
    g .= x-prox(x-steplength*g, steplength)

    # Nesterov momentum
    if Nesterov
        t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2))
        g .*= (t+t0-T(1))/t
        opt.t = t
    end

    # Update counter
    ~isnothing(opt.counter) && (opt.counter += 1)
    ~isnothing(reset_counter) && (opt.counter > reset_counter) && reset!(opt)

    return g

end


# Other utils

function spectral_radius(A::AT, x::AbstractArray{T,N}; niter::Int64=10) where {T,N,AT<:Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}}
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


function linesearch_backtracking(obj::Function, x0::AbstractArray{CT,N}, p::AbstractArray{CT,N}, lr::T; fx0::Union{Nothing,T}=nothing, niter::Integer=3, mult_fact::T=T(0.5), verbose::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T}}

    isnothing(fx0) && (fx0 = obj(x0))
    for _ = 1:niter
        fx = obj(x0+lr*p)
        fx < fx0 ? break : (verbose && print("."); lr *= mult_fact)
    end
    return lr

end