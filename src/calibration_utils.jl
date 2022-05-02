# Calibration operator utilities

export GlobalCalibration, GlobalCalibrationOperator, ReadoutCalibration, ReadoutCalibrationOperator
export calibration, calibration_linop

abstract type AbstractCalibrationOperator{CT}<:AbstractLinearOperator{CT,2,2} end
abstract type AbstractCalibration{T<:Real} end


## Calibration types

struct ReadoutCalibration{T}<:AbstractCalibration{T}
    λ::T
end

struct ReadoutCalibrationOnlyPhase{T}<:AbstractCalibration{T}
    λ::T
end

struct ReadoutCalibrationOnlyAmplitude{T}<:AbstractCalibration{T}
    λ::T
end

struct GlobalCalibration{T}<:AbstractCalibration{T}
    λ::T
end

function calibration(type::Symbol, λ::T) where {T<:Real}
    (type == :readout)           && (return ReadoutCalibration{T}(λ))
    (type == :readout_phase)     && (return ReadoutCalibrationOnlyPhase{T}(λ))
    (type == :readout_amplitude) && (return ReadoutCalibrationOnlyAmplitude{T}(λ))
    (type == :global)            && (return GlobalCalibration{T}(λ))
end


## Readout-regularized calibration

struct ReadoutCalibrationOperator{CT<:Complex}<:AbstractCalibrationOperator{CT}
    α::AbstractVector{CT}
    nk::Integer
end

function (C::ReadoutCalibration{T})(d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}) where {T<:Real,CT<:Complex{T}}
    normd2 = sum(abs.(d).^2; dims=2)
    α = vec((sum(conj(d̄).*d; dims=2)+C.λ^2*normd2)./(sum(abs.(d̄).^2; dims=2).+C.λ^2*normd2))
    return ReadoutCalibrationOperator{CT}(α, size(d,2))
end

function (C::ReadoutCalibrationOnlyPhase{T})(d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}) where {T<:Real,CT<:Complex{T}}
    normd2 = sum(abs.(d).^2; dims=2)
    α = vec((sum(conj(d̄).*d; dims=2)+C.λ^2*normd2)./(sum(abs.(d̄).^2; dims=2).+C.λ^2*normd2))
    # abs_α = abs.(α)
    # idx = abs_α .> T(1)
    # α[idx] .= α[idx]./abs.(α[idx])
    α ./= abs.(α)
    return ReadoutCalibrationOperator{CT}(α, size(d,2))
end

function (C::ReadoutCalibrationOnlyAmplitude{T})(d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}) where {T<:Real,CT<:Complex{T}}
    normd2 = sum(abs.(d).^2; dims=2)
    α = min.(max.(vec(real(sum(conj(d̄).*d; dims=2)+C.λ^2*normd2)./(sum(abs.(d̄).^2; dims=2).+C.λ^2*normd2)), T(0)), T(1))
    return ReadoutCalibrationOperator{CT}(α, size(d,2))
end

AbstractLinearOperators.domain_size(A::ReadoutCalibrationOperator) = (length(A.α), A.nk)
AbstractLinearOperators.range_size(A::ReadoutCalibrationOperator) = domain_size(A)
AbstractLinearOperators.matvecprod(A::ReadoutCalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex} = A.α.*r
AbstractLinearOperators.matvecprod_adj(A::ReadoutCalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex} = conj(A.α).*r

struct GlobalCalibrationOperator{CT<:Complex}<:AbstractCalibrationOperator{CT}
    α::CT
    nt::Integer
    nk::Integer
end

(C::GlobalCalibration{T})(d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}) where {T<:Real,CT<:Complex{T}} = GlobalCalibrationOperator{CT}((dot(d̄, d)+C.λ^2*norm(d)^2)/(norm(d̄)^2+C.λ^2*norm(d)^2), size(d)...)

AbstractLinearOperators.domain_size(A::GlobalCalibrationOperator) = (A.nt, A.nk)
AbstractLinearOperators.range_size(A::GlobalCalibrationOperator) = domain_size(A)
AbstractLinearOperators.matvecprod(A::GlobalCalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex} = A.α*r
AbstractLinearOperators.matvecprod_adj(A::GlobalCalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex} = conj(A.α)*r


## Other utils

LinearAlgebra.norm(A::AT) where {AT<:Union{ReadoutCalibrationOperator,GlobalCalibrationOperator}} = norm(A.α, Inf)


# ## General calibration operator

# calibration_linop(d̄::AbstractArray{CT,2}, d::AbstractArray{CT,2}, opt::CalibrationOptions{T}; W::AbstractLinearOperator{CT,2,2}=identity_operator(CT, size(d))) where {T<:Real,CT<:Complex{T}} = calibration_linop(d̄, d, opt.λ; W=W, readout_reg=opt.readout_reg)

# struct CalibrationOperator{CT}<:AbstractCalibrationOperatorReadout{CT}
#     d̄::AbstractArray{CT,2}
#     d::AbstractArray{CT,2}
#     λ::Real
#     W::AbstractLinearOperator{CT,2,2}
# end

# AbstractLinearOperators.domain_size(A::CalibrationOperator) = size(A.d)
# AbstractLinearOperators.range_size(A::CalibrationOperator) = domain_size(A)
# function AbstractLinearOperators.matvecprod(A::CalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex}
#     d̄ = A.W*A.d̄; d = A.W*A.d
#     r_ = A.W*r
#     r_ = r_-d̄*dot(d̄, r_)/(norm(d̄)^2+A.λ^2)
#     r_ .+= d*dot(d̄, r_)/A.λ^2
#     return A.W\r_
# end
# function AbstractLinearOperators.matvecprod_adj(A::CalibrationOperator{CT}, r::AbstractArray{CT,2}) where {CT<:Complex}
#     d̄ = A.W*A.d̄; d = A.W*A.d
#     r_ = A.W\r
#     r_ = r_+d̄*dot(d, r_)/A.λ^2
#     r_ .-= d̄*dot(d̄, r_)/(norm(d̄)^2+A.λ^2)
#     return A.W*r_
# end