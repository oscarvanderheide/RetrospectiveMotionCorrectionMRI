#: Projectionable sets

export ZeroSet, zero_set
export SublevelSet


## Homogeneous constraints via masking grid function

struct ZeroSet{T,N}<:AbstractProjectionableSet{T,N}
    is_zero::AbstractArray{Bool,N}
end

zero_set(T::DataType, is_zero::AbstractArray{Bool,N}) where N = ZeroSet{T,N}(is_zero)

Base.in(x::AbstractArray{T,N}, C::ZeroSet{T,N}) where {T,N} = all(x[C.is_zero] .== 0)

function proj!(x::AbstractArray{T,N}, C::ZeroSet{T,N}, ::ExactArgmin, y::AbstractArray{T,N}) where {T,N}
    idx_nonzero = (!).(C.is_zero)
    y[idx_nonzero] .= x[idx_nonzero]
    y[C.is_zero] .= 0
    return y
end


## Sub-level sets with proximable functions

struct SublevelSet{T,N}<:AbstractProjectionableSet{T,N}
    fun::AbstractProximableFunction{T,N}
    level::Real
end

Base.:≤(g::AbstractProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = SublevelSet{CT,N}(g, ε)

Base.in(x::AbstractArray{T,N}, C::SublevelSet{T,N}) where {T<:RealOrComplex,N} = C.fun(x) ≤ C.level

proj!(x::AbstractArray{T,N}, C::SublevelSet{T,N}, options::AbstractArgminOptions, y::AbstractArray{T,N}) where {T,N} = proj!(x, C.level, C.fun, options, y)

options(C::SublevelSet) = options(C.fun)