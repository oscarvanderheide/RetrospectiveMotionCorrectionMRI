#: Fused (structurally weighted) gradient operator

export WeightedGradientOperator, weighted_gradient_operator


# Weighted gradient: u -> P*∇u, with P an (optional) projection on a vector field
#
# Equivalent to `weight*gradient_operator(n, h)`, but evaluated in a single pass over the
# data (forward differences + pointwise projection) without intermediate arrays. Multithreaded
# kernels are used for 3D CPU arrays; other array types fall back to broadcasting.

struct WeightedGradientOperator{T,N,M}<:AbstractLinearOperator{T,N,T,M}
    n::NTuple{N,Int64}
    h::NTuple{N,Any}
    weight::Union{Nothing,ProjVectorField{T,M}}
    buffer::Base.RefValue{Any} # workspace for the adjoint
end

"""
    weighted_gradient_operator(T, n, h; weight=nothing)

Returns the linear operator ``\\mathbf{u}\\mapsto P\\nabla\\mathbf{u}`` (forward differences) for element type `T`, grid size `n` and spacing `h`. `weight` is either `nothing` (``P=I``) or a `ProjVectorField` as returned by [`structural_weight`](@ref).
"""
function weighted_gradient_operator(::Type{CT}, n::NTuple{N,Int64}, h::NTuple{N,T}; weight::Union{Nothing,ProjVectorField}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ~isnothing(weight) && (size(weight.ξ) != ((n.-1)..., N)) && throw(ArgumentError("Weight size $(size(weight.ξ)) not consistent with gradient range $(((n.-1)..., N))"))
    return WeightedGradientOperator{CT,N,N+1}(n, h, weight, Ref{Any}(nothing))
end

AbstractLinearOperators.domain_size(A::WeightedGradientOperator) = A.n
AbstractLinearOperators.range_size(A::WeightedGradientOperator{T,N}) where {T,N} = ((A.n.-1)..., N)
AbstractLinearOperators.label(A::WeightedGradientOperator) = isnothing(A.weight) ? "∇" : "P∇"

AbstractLinearOperators.matvecprod(A::WeightedGradientOperator{T,N,M}, u::AbstractArray{T,N}) where {T,N,M} = weighted_gradient!(similar(u, range_size(A)), A, u)
AbstractLinearOperators.matvecprod_adj(A::WeightedGradientOperator{T,N,M}, p::AbstractArray{T,M}) where {T,N,M} = weighted_gradient_adj!(similar(p, domain_size(A)), A, p)


## Generic fallback (broadcasting)

function weighted_gradient!(p::AbstractArray{T,M}, A::WeightedGradientOperator{T,N,M}, u::AbstractArray{T,N}) where {T,N,M}
    p .= gradient_eval(u, convert_spacing(T, A.h))
    ~isnothing(A.weight) && (p .= A.weight*p)
    return p
end

function weighted_gradient_adj!(u::AbstractArray{T,N}, A::WeightedGradientOperator{T,N,M}, p::AbstractArray{T,M}) where {T,N,M}
    isnothing(A.weight) ? (q = p) : (q = A.weight*p)
    h = convert_spacing(T, A.h)
    n = A.n
    u .= 0
    idx = ntuple(k -> 1:n[k]-1, N)
    @inbounds for d = 1:N
        idx_p1 = ntuple(k -> (k == d) ? (2:n[k]) : (1:n[k]-1), N)
        qd = selectdim(q, M, d)
        view(u, idx...)    .-= qd./h[d]
        view(u, idx_p1...) .+= qd./h[d]
    end
    return u
end

convert_spacing(::Type{T}, h::NTuple{N,Any}) where {T,N} = ntuple(i -> real(T)(h[i]), N)


## Fused multithreaded kernels (3D CPU arrays)

function weighted_gradient!(p::Array{T,4}, A::WeightedGradientOperator{T,3,4}, u::Array{T,3}) where {T}
    h = convert_spacing(T, A.h)
    isnothing(A.weight) && (return gradient_kernel!(p, u, h))
    ξ = A.weight.ξ
    ξ isa Array{T,4} || return invoke(weighted_gradient!, Tuple{AbstractArray{T,4},WeightedGradientOperator{T,3,4},AbstractArray{T,3}}, p, A, u)
    return weighted_gradient_kernel!(p, u, ξ, A.weight.γ, h)
end

function weighted_gradient_adj!(u::Array{T,3}, A::WeightedGradientOperator{T,3,4}, p::Array{T,4}) where {T}
    h = convert_spacing(T, A.h)
    isnothing(A.weight) && (return neg_divergence_kernel!(u, p, h))
    ξ = A.weight.ξ
    ξ isa Array{T,4} || return invoke(weighted_gradient_adj!, Tuple{AbstractArray{T,3},WeightedGradientOperator{T,3,4},AbstractArray{T,4}}, u, A, p)
    q = A.buffer[]
    (q isa Array{T,4} && size(q) == size(p)) || (q = similar(p); A.buffer[] = q)
    projection_kernel!(q, p, ξ, A.weight.γ) # P is self-adjoint
    return neg_divergence_kernel!(u, q, h)
end

function gradient_kernel!(p::Array{T,4}, u::Array{T,3}, (h1, h2, h3)::NTuple{3,<:Real}) where {T}
    nx, ny, nz = size(u)
    Threads.@threads for k = 1:nz-1
        @inbounds for j = 1:ny-1, i = 1:nx-1
            c = u[i,j,k]
            p[i,j,k,1] = (u[i+1,j,k]-c)/h1
            p[i,j,k,2] = (u[i,j+1,k]-c)/h2
            p[i,j,k,3] = (u[i,j,k+1]-c)/h3
        end
    end
    return p
end

function weighted_gradient_kernel!(p::Array{T,4}, u::Array{T,3}, ξ::Array{T,4}, γ::T, (h1, h2, h3)::NTuple{3,<:Real}) where {T}
    nx, ny, nz = size(u)
    Threads.@threads for k = 1:nz-1
        @inbounds for j = 1:ny-1, i = 1:nx-1
            c = u[i,j,k]
            g1 = (u[i+1,j,k]-c)/h1
            g2 = (u[i,j+1,k]-c)/h2
            g3 = (u[i,j,k+1]-c)/h3
            ξ1 = ξ[i,j,k,1]; ξ2 = ξ[i,j,k,2]; ξ3 = ξ[i,j,k,3]
            s = γ*(g1*conj(ξ1)+g2*conj(ξ2)+g3*conj(ξ3))
            p[i,j,k,1] = g1-ξ1*s
            p[i,j,k,2] = g2-ξ2*s
            p[i,j,k,3] = g3-ξ3*s
        end
    end
    return p
end

function projection_kernel!(q::Array{T,4}, p::Array{T,4}, ξ::Array{T,4}, γ::T) where {T}
    nx, ny, nz, _ = size(p)
    Threads.@threads for k = 1:nz
        @inbounds for j = 1:ny, i = 1:nx
            p1 = p[i,j,k,1]; p2 = p[i,j,k,2]; p3 = p[i,j,k,3]
            ξ1 = ξ[i,j,k,1]; ξ2 = ξ[i,j,k,2]; ξ3 = ξ[i,j,k,3]
            s = γ*(p1*conj(ξ1)+p2*conj(ξ2)+p3*conj(ξ3))
            q[i,j,k,1] = p1-ξ1*s
            q[i,j,k,2] = p2-ξ2*s
            q[i,j,k,3] = p3-ξ3*s
        end
    end
    return q
end

# Adjoint of forward differences (q lives on the (n.-1) grid)
function neg_divergence_kernel!(u::Array{T,3}, q::Array{T,4}, (h1, h2, h3)::NTuple{3,<:Real}) where {T}
    nx, ny, nz = size(u)
    Threads.@threads for k = 1:nz
        @inbounds for j = 1:ny, i = 1:nx
            in_i = i < nx; in_j = j < ny; in_k = k < nz
            acc = zero(T)
            (in_i && in_j && in_k) && (acc -= q[i,j,k,1]/h1+q[i,j,k,2]/h2+q[i,j,k,3]/h3)
            (i > 1 && in_j && in_k) && (acc += q[i-1,j,k,1]/h1)
            (j > 1 && in_i && in_k) && (acc += q[i,j-1,k,2]/h2)
            (k > 1 && in_i && in_j) && (acc += q[i,j,k-1,3]/h3)
            u[i,j,k] = acc
        end
    end
    return u
end
