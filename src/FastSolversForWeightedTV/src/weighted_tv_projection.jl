#: Fast projection on the (structurally weighted) TV ball, for 3D arrays (CPU or GPU)
#
# Specialization of `proj!(y, ε, g::WeightedProximableFunction, options::ArgminFISTA, x)` for
# g(u) = ||P∇u||_{2,1} (as returned by `gradient_norm(2, 1, ...)`). It performs exactly the same
# dual FISTA iterations as the generic implementation in AbstractProximableFunctions:
#
#   min_p 1/2||(P∇)'p-y||^2+δ*_{||.||_{2,1}≤ε}(p),    x = y-(P∇)'p,
#
# but with fused kernels (multithreaded loops for CPU arrays, broadcasting otherwise, e.g. on GPU),
# preallocated workspaces, and an exact finite-step (Michelot) computation of the L21-ball
# projection threshold instead of a Brent root search.
# Optionally, the dual variable is warm-started from the previous call (see `gradient_norm`).


function AbstractProximableFunctions.weighted_proj!(y::AbstractArray{CT,3}, ε::Real, g::WeightedProximableFunction, A::WeightedGradientOperator{CT,3,4}, ::ProximableMixedNorm{CT,4,2,1}, options::ArgminFISTA, x::AbstractArray{CT,3}) where {CT<:RealOrComplex}
    # Weight and input must live on the same device
    ~isnothing(A.weight) && ~(A.weight.ξ isa typeof(similar(y, CT, size(A.weight.ξ)))) && return AbstractProximableFunctions.proj_weighted_dual_fista!(y, ε, g, options, x)
    p = weighted_tv_dual_fista(A, y, real(CT)(ε), options)
    weighted_gradient_adj!(x, A, p)
    return x .= y.-x
end

function weighted_tv_dual_fista(A::WeightedGradientOperator{CT,3,4}, y::AbstractArray{CT,3}, ε::T, options::ArgminFISTA) where {T<:Real,CT<:RealOrComplex{T}}

    # Workspaces
    p     = fill!(similar(y, CT, range_size(A)), 0) # current (extrapolated) iterate
    p_    = similar(p)                              # proximal step
    pprev = fill!(similar(p), 0)
    if A.warmstart && (A.dual[] isa typeof(p)) && (size(A.dual[]) == size(p))
        copyto!(p, A.dual[]); copyto!(pprev, p)
    end
    G     = similar(p)               # gradient
    r     = similar(y)               # residual (P∇)'p-y
    ptn   = similar(y, T, size(p)[1:3])

    L = T(options.Lipschitz_constant)
    counter = isnothing(options.reset_counter) ? nothing : 0
    t0 = T(1)
    fhist = options.fun_history

    for n = 1:options.niter

        # Gradient: (P∇)((P∇)'p-y)
        weighted_gradient_adj!(r, A, p)
        tmap!((ri, yi) -> ri-yi, r, r, y)
        weighted_gradient!(G, A, r)
        if options.verbose || ~isnothing(fhist)
            fval_n = norm(r)^2/2
            ~isnothing(fhist) && (fhist[n] = fval_n)
            options.verbose && (@info string("Iter: ", n, ", fval: ", fval_n))
        end

        # Proximal step for the conjugate of the L21-ball indicator (Moreau):
        #   p_ = z-1/L*proj(L*z), with z = p-G/L
        conjugate_ball_prox!(p_, p, G, L, ε, ptn)

        # Nesterov acceleration
        if options.Nesterov
            t = (1+sqrt(1+4*t0^2))/2
            β = T((t0-1)/t)
            (n == 1) ? copyto!(p, p_) : let β = β; tmap!((a, b) -> a+β*(a-b), p, p_, pprev) end
            ~isnothing(counter) && (counter += 1)
            t0 = t
            copyto!(pprev, p_)
            ~isnothing(options.reset_counter) && (counter >= options.reset_counter) && (t0 = T(1))
        else
            copyto!(p, p_)
        end

    end

    A.warmstart && (A.dual[] = p)
    return p

end

function conjugate_ball_prox!(p_::Array{CT,4}, p::Array{CT,4}, G::Array{CT,4}, L::T, ε::T, ptn::Array{T,3}) where {T<:Real,CT<:RealOrComplex{T}}
    η2 = 3*eps(T)^2 # consistent with ptnorm2(.; η=eps(T)) in the generic L21 projection
    nx, ny, nz, _ = size(p)

    # Pointwise norms of w = L*z
    Threads.@threads for k = 1:nz
        @inbounds for j = 1:ny, i = 1:nx
            w1 = L*(p[i,j,k,1]-G[i,j,k,1]/L); w2 = L*(p[i,j,k,2]-G[i,j,k,2]/L); w3 = L*(p[i,j,k,3]-G[i,j,k,3]/L)
            ptn[i,j,k] = sqrt(abs2(w1)+abs2(w2)+abs2(w3)+η2)
        end
    end

    # Projection threshold λ (λ = 0 if w already in the ball)
    λ = l1ball_threshold(ptn, ε)

    # p_ = z-proj(w)/L, with proj(w) = max(1-λ/ptn, 0)*w
    Threads.@threads for k = 1:nz
        @inbounds for j = 1:ny, i = 1:nx
            s = (ptn[i,j,k] >= λ) ? (1-λ/ptn[i,j,k]) : zero(T)
            for d = 1:3
                z = p[i,j,k,d]-G[i,j,k,d]/L
                p_[i,j,k,d] = z-s*(L*z)/L
            end
        end
    end
    return p_
end

function conjugate_ball_prox!(p_::AbstractArray{CT,4}, p::AbstractArray{CT,4}, G::AbstractArray{CT,4}, L::T, ε::T, ptn::AbstractArray{T,3}) where {T<:Real,CT<:RealOrComplex{T}}
    η2 = 3*eps(T)^2
    z = p.-G./L
    ptn .= dropdims(sqrt.(sum(abs2.(L.*z); dims=4).+η2); dims=4)
    λ = l1ball_threshold(ptn, ε)
    s = ifelse.(ptn .>= λ, 1 .-λ./ptn, zero(T))
    p_ .= z.-reshape(s, size(s)..., 1).*(L.*z)./L
    return p_
end

"""
    l1ball_threshold(a, ε)

For nonnegative `a`, returns the λ≥0 such that `sum(max.(a.-λ, 0)) == ε` (or 0 when `sum(a) ≤ ε`), i.e. the soft-thresholding level of the projection onto the L1 ball of radius `ε`. Exact, via Michelot's finite-step algorithm (each step is a threaded pass over `a`).
"""
function l1ball_threshold(a::AbstractArray{T}, ε::T) where {T<:Real}
    s, c = sum_count_above(a, T(-Inf))
    (s <= ε) && (return T(0))
    λ = (s-ε)/c
    for _ = 1:10_000
        s, c_new = sum_count_above(a, T(λ))
        λ_new = (s-ε)/c_new
        (c_new == c || λ_new <= λ) && (λ = max(λ, λ_new); break)
        λ, c = λ_new, c_new
    end
    return T(λ)
end

function sum_count_above(a::Array{T}, λ::T) where {T<:Real}
    nchunks = Threads.nthreads()
    sums = zeros(Float64, nchunks); counts = zeros(Int, nchunks)
    N = length(a)
    Threads.@threads :static for c = 1:nchunks
        s = 0.0; m = 0
        @inbounds for i = (c-1)*N÷nchunks+1:c*N÷nchunks
            ai = a[i]
            (ai > λ) && (s += ai; m += 1)
        end
        sums[c] = s; counts[c] = m
    end
    return sum(sums), sum(counts)
end

# (single reduction, i.e. one device synchronization per Michelot step on GPUs)
function sum_count_above(a::AbstractArray{T}, λ::T) where {T<:Real}
    s, c = mapreduce(x -> ifelse(x > λ, (x, 1), (zero(T), 0)), (v, w) -> (v[1]+w[1], v[2]+w[2]), a; init=(zero(T), 0))
    return Float64(s), c
end

# Elementwise map (threaded for CPU arrays)
tmap!(f, out::AbstractArray, args::AbstractArray...) = (out .= f.(args...))
# Threaded elementwise map (CPU arrays)
function tmap!(f, out::Array, args::Array...)
    N = length(out)
    nchunks = Threads.nthreads()
    Threads.@threads :static for c = 1:nchunks
        @inbounds for i = (c-1)*N÷nchunks+1:c*N÷nchunks
            out[i] = f(getindex.(args, i)...)
        end
    end
    return out
end
