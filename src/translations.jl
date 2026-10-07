# Translation utilities

export PhaseShiftLinOp, JacobianPhaseShift, PhaseShiftParametericLinOp, PhaseShiftParametericDelayedEval
export phase_shift_linop, phase_shift, Jacobian, ∂


## Linear operator

struct PhaseShiftLinOp{T<:Real}<:AbstractLinearOperator{Complex{T},2,2}
    phase_shift::AbstractArray{Complex{T},2}
end

phase_shift_linop(k::AbstractArray{T,3}, τ::AbstractArray{T,2}) where {T<:Real} = PhaseShiftLinOp{T}(phase_shift(k, τ))

phase_shift(k::AbstractArray{T,3}, τ::AbstractArray{T,2}) where {T<:Real} = exp.(-im*(k[:,:,1].*τ[:,1]+k[:,:,2].*τ[:,2]+k[:,:,3].*τ[:,3]))

AbstractLinearOperators.domain_size(P::PhaseShiftLinOp) = size(P.phase_shift)
AbstractLinearOperators.range_size(P::PhaseShiftLinOp) = domain_size(P)

AbstractLinearOperators.matvecprod(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} = P.phase_shift.*d
Base.:*(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},3}) where {T<:Real} = P.phase_shift.*d
AbstractLinearOperators.matvecprod_adj(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} = conj(P.phase_shift).*d


## Parameteric linear operator

struct PhaseShiftParametericLinOp{T<:Real}
    kcoord::AbstractArray{T,3}
end

phase_shift(K::AbstractArray{T,3}) where {T<:Real} = PhaseShiftParametericLinOp(K)

(P::PhaseShiftParametericLinOp{T})(τ::AbstractArray{T,2}) where {T<:Real} = phase_shift_linop(P.kcoord, τ)

(P::PhaseShiftParametericLinOp)() = P


## Parameteric delayed evaluation

struct PhaseShiftParametericDelayedEval{T<:Real}
    P::PhaseShiftParametericLinOp{T}
    d::AbstractArray{Complex{T},2}
end

Base.:*(P::PhaseShiftParametericLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} =  PhaseShiftParametericDelayedEval{T}(P, d)

(Pd::PhaseShiftParametericDelayedEval{T})(τ::AbstractArray{T,2}) where {T<:Real} = phase_shift_linop(Pd.P.kcoord, τ)*Pd.d


## Jacobian of phase-shift evaluated

struct JacobianPhaseShift{T<:Real}<:AbstractLinearOperator{Complex{T},2,2}
    ∂P::AbstractArray{Complex{T},3}
end

function Jacobian(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real}
    Pτ = phase_shift_linop(Pd.P.kcoord, τ)
    Pτd = Pτ*Pd.d
    nt, nk, _ = size(Pd.P.kcoord)
    J = -im*Pd.P.kcoord.*reshape(Pτd,nt,nk,1)
    return Pτd, Pτ, JacobianPhaseShift{T}(J)
end

∂(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Pd, τ)

AbstractLinearOperators.domain_size(∂Pd::JacobianPhaseShift) = (size(∂Pd.∂P,1),3)
AbstractLinearOperators.range_size(∂Pd::JacobianPhaseShift) = prod(size(∂Pd.∂P)[1:2])

AbstractLinearOperators.matvecprod(∂Pd::JacobianPhaseShift{T}, Δτ::AbstractArray{CT,2}) where {T<:Real,CT<:RealOrComplex{T}} = ∂Pd.∂P[:,:,1].*Δτ[:,1]+∂Pd.∂P[:,:,2].*Δτ[:,2]+∂Pd.∂P[:,:,3].*Δτ[:,3]
Base.:*(∂Pd::JacobianPhaseShift{T}, Δτ::AbstractArray{T,2}) where {T<:Real} = AbstractLinearOperators.matvecprod(∂Pd, Δτ)
AbstractLinearOperators.matvecprod_adj(∂Pd::JacobianPhaseShift{T}, Δd::AbstractArray{Complex{T},2}) where {T<:Real} = real(sum(conj(∂Pd.∂P).*Δd; dims=2)[:,1,:])