# Optimization utilities:
# - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
# - Anderson acceleration: Wei et al., 2021, Stochastic Anderson Mixing for Nonconvex Stochastic Optimization

export FISTA, FISTA!, Anderson
export reset!, linesearch_backtracking, spectral_radius


## FISTA

abstract type AbstractFISTA{T}<:Flux.Optimise.AbstractOptimiser end

mutable struct FISTA{T}<:AbstractFISTA{T}
    L::T
    prox::Function
    t::T
    Nesterov::Bool
end

mutable struct FISTA!{T}<:AbstractFISTA{T}
    L::T
    prox!::Function
    t::T
    Nesterov::Bool
end

FISTA(L::T, prox::Function; t::T=T(1), Nesterov::Bool=true) where {T<:Real} = FISTA{T}(L, prox, t, Nesterov)

FISTA!(L::T, prox!::Function; t::T=T(1), Nesterov::Bool=true) where {T<:Real} = FISTA!{T}(L, prox!, t, Nesterov)

reset!(opt::AbstractFISTA{T}) where {T<:Real} = (opt.t = T(1))

function Flux.Optimise.apply!(opt::AbstractFISTA{T}, x::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Gradient + proxy update
    steplength = T(1)/opt.L
    if opt isa FISTA
        g .= x-opt.prox(x-steplength*g, steplength)
    elseif opt isa FISTA!
        opt.prox!(x-steplength*g, steplength, g); g .= x.-g
    end

    # Nesterov momentum
    if opt.Nesterov
        t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*opt.t^2))
        g .*= (t+opt.t-T(1))/t
        opt.t = t
    end

    return g

end

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


## Anderson acceleration

mutable struct Anderson{T}<:Flux.Optimise.AbstractOptimiser
    lr::Union{Nothing,T}
    hist_size::Integer
    β::T
    ΔX::Union{Nothing,Vector{<:AbstractArray}}
    ΔG::Union{Nothing,Vector{<:AbstractArray}}
    x0::Union{Nothing,AbstractArray}
    g0::Union{Nothing,AbstractArray}
    init::Bool
end

Anderson(; lr::Union{Nothing,T}=nothing, hist_size::Integer=5, β::T=1.0) where {T<:Real} = Anderson{T}(lr, hist_size, β, nothing, nothing, nothing, nothing, false)

reset!(opt::Anderson) = (opt.ΔX = nothing; opt.ΔG = nothing; opt.x0 = nothing; opt.g0 = nothing; opt.init = false)

function Flux.Optimise.apply!(opt::Anderson{T}, x::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Rescaling gradient w/ learning rate
    ~isnothing(opt.lr) && (g .*= opt.lr)

    # Initialize if necessary
    if ~opt.init
        opt.ΔX = Vector{typeof(x)}(undef,0)
        opt.ΔG = Vector{typeof(g)}(undef,0)
        opt.x0 = deepcopy(x)
        opt.g0 = deepcopy(g)
        opt.init = true
        return g
    end

    # Update model/residual history
    if length(opt.ΔX) < opt.hist_size
        push!(opt.ΔX, x-opt.x0)
        push!(opt.ΔG, g-opt.g0)
    else
        for i = 1:opt.hist_size-1
            opt.ΔX[i] .= opt.ΔX[i+1]
            opt.ΔG[i] .= opt.ΔG[i+1]
        end
        opt.ΔX[end] .= x-opt.x0
        opt.ΔG[end] .= g-opt.g0
    end
    opt.x0 .= x
    opt.g0 .= g
    
    # Computing steplength
    Γ = solve_Anderson_steplength(opt.ΔG, g)
    
    # Gradient update
    return g .= sum(opt.ΔX.*Γ)+opt.β*(g-sum(opt.ΔG.*Γ))

end

function solve_Anderson_steplength(ΔR::AbstractArray{AT,1}, r::AbstractArray{CT}) where {T<:Real,CT<:RealOrComplex{T},AT<:AbstractArray{CT}}
    n = length(ΔR)
    nx = prod(size(ΔR[1]))
    A = Array{CT,2}(undef,nx,n)
    @inbounds for j = 1:n
        A[:,j] = vec(ΔR[j])
    end
    return A\vec(r)
end


function linesearch_backtracking(obj::Function, x0::AbstractArray{CT,N}, p::AbstractArray{CT,N}, lr::T; fx0::Union{Nothing,T}=nothing, niter::Integer=3, mult_fact::T=T(0.5), verbose::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T}}

    isnothing(fx0) && (fx0 = obj(x0))
    for n = 1:niter
        fx = obj(x0+lr*p)
        fx < fx0 ? break : (verbose && print("."); lr *= mult_fact)
    end
    return lr

end