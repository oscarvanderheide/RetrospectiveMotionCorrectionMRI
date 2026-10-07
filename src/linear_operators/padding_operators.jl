export AbstractPaddingOperator,
       ZeroPaddingOperator, zero_padding_operator,
       RepeatPaddingOperator, repeat_padding_operator,
       extended_size


# Padding operators

abstract type AbstractPaddingOperator{T,N}<:AbstractLinearOperator{T,N,T,N} end


## Zero padding

struct ZeroPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    padding
end

zero_padding_operator(T::DataType, padding::NTuple{M,Integer}) where M = (mod(M,2) == 0) ? ZeroPaddingOperator{T,div(M,2)}(padding) : throw(ArgumentError("Padding size not consistent"))
zero_padding_operator(T::DataType, N::Integer, padding::Integer) = ZeroPaddingOperator{T,N}(padding)

label(::AbstractPaddingOperator) = "Padding"

function matvecprod(P::ZeroPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    ext_size = extended_size(size(u), P.padding)
    cview = center_view(ext_size, P.padding)
    u_ext = zeros(T, ext_size)
    view(u_ext, cview...) .= u
    return u_ext
end

function matvecprod_adj(P::ZeroPaddingOperator{T,N}, u_ext::AbstractArray{T,N}) where {T,N}
    ext_size = size(u_ext)
    cview = center_view(ext_size, P.padding)
    return u_ext[cview...]
end

extended_size(size::NTuple{N,Integer}, padding::NTuple) where N = Tuple([size[i]+sum(padding[2*(i-1)+1:2*i]) for i=1:N])
extended_size(size::NTuple{N,Integer}, padding::Integer) where N = size.+2*padding
reduced_size(ext_size::NTuple{N,Integer}, padding::NTuple) where N = Tuple([ext_size[i]-sum(padding[2*(i-1)+1:2*i]) for i=1:N])
reduced_size(ext_size::NTuple{N,Integer}, padding::Integer) where N = ext_size.-2*padding
center_view(ext_size::NTuple{N,Integer}, padding::NTuple) where N = Tuple([padding[2*(i-1)+1]+1:ext_size[i]-padding[2*i] for i = 1:N])
center_view(ext_size::NTuple{N,Integer}, padding::Integer) where N = Tuple([padding+1:ext_size[i]-padding for i = 1:N])

function full_matrix(P::ZeroPaddingOperator{T,N}, input_size::NTuple{N,Integer}) where {T,N}
    ext_size = extended_size(input_size, P.padding)
    cview = center_view(ext_size, P.padding)
    I = vec(LinearIndices(ext_size)[CartesianIndices(cview)])
    return sparse(I, 1:prod(input_size), ones(T, length(I)), prod(ext_size), prod(input_size))
end


## Repeat padding

struct RepeatPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    padding
end

repeat_padding_operator(T::DataType, padding::NTuple{M,Integer}) where M = (mod(M,2) == 0) ? RepeatPaddingOperator{T,div(M,2)}(padding) : throw(ArgumentError("Padding size not consistent"))
repeat_padding_operator(T::DataType, N::Integer, padding::Integer) = RepeatPaddingOperator{T,N}(padding)

function matvecprod(P::RepeatPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}

    padding = P.padding
    ext_size = extended_size(size(u), P.padding)
    cview = center_view(ext_size, P.padding)
    u_ext = similar(u, ext_size)
    view(u_ext, cview...) .= u
    @inbounds for i = 1:N
        selectdim(u_ext, i, 1:padding[(i-1)*2+1]) .= selectdim(u_ext, i, padding[(i-1)*2+1]+1:padding[(i-1)*2+1]+1)
        selectdim(u_ext, i, ext_size[i]-padding[2*i]+1:ext_size[i]) .= selectdim(u_ext, i, ext_size[i]-padding[2*i]:ext_size[i]-padding[2*i])
    end
    return u_ext

end

function matvecprod_adj(P::RepeatPaddingOperator{T,N}, u_ext::AbstractArray{T,N}) where {T,N}

    padding = P.padding
    ext_size = size(u_ext)
    u = copy(u_ext)
    @inbounds for i = 1:N
        u_top = sum(selectdim(u, i, 1:padding[(i-1)*2+1]); dims=i)
        u_bottom = sum(selectdim(u, i, ext_size[i]-padding[2*i]+1:ext_size[i]); dims=i)
        u = selectdim(u, i, padding[(i-1)*2+1]+1:ext_size[i]-padding[2*i])
        selectdim(u, i, 1:1) .+= u_top
        selectdim(u, i, size(u, i):size(u, i)) .+= u_bottom
    end
    return copy(u)

end