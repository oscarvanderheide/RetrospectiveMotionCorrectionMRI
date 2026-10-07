export AbstractConvolutionOperator, ConvolutionOperator, convolution_operator

abstract type AbstractConvolutionOperator{T,N}<:AbstractLinearOperator{T,N,T,N} end


# Convolution operator

mutable struct ConvolutionOperator{T,N}<:AbstractConvolutionOperator{T,N}
    stencil::AbstractArray{T,N}
    stride
    padding
    dilation
    flipped::Bool
    groups
    cdims::Union{Nothing,DenseConvDims}
    cdims_onthefly::Bool
end

convolution_operator(
    stencil::AbstractArray{T,N};
    stride=1, padding=0, dilation=1, flipped::Bool=false, groups=1,
    cdims::Union{Nothing,DenseConvDims}=nothing, cdims_onthefly::Bool=true) where {T,N} =
    ConvolutionOperator{T,N}(stencil, stride, padding, dilation, flipped, groups, cdims, cdims_onthefly)

AbstractLinearOperators.domain_size(C::ConvolutionOperator{T,N}) where {T,N} = ~is_init(C) ? Tuple(Vector{Nothing}(undef,N)) : input_dims(C.cdims)
AbstractLinearOperators.range_size(C::ConvolutionOperator{T,N}) where {T,N} = ~is_init(C) ? Tuple(Vector{Nothing}(undef,N)) : output_dims(C.cdims)
AbstractLinearOperators.label(::ConvolutionOperator) = "Conv"

input_dims(cdims::DenseConvDims)  = Tuple([NNlib.input_size(cdims)...,  NNlib.channels_in(cdims),  "nb"])
output_dims(cdims::DenseConvDims) = Tuple([NNlib.output_size(cdims)..., NNlib.channels_out(cdims), "nb"])

"""
Size of input is WHCN (width, height(, depth), channel, batch)
"""
function AbstractLinearOperators.matvecprod(C::ConvolutionOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    ~is_init(C) && initialize!(C, u)
    return conv(u, C.stencil, C.cdims)
end
AbstractLinearOperators.matvecprod_adj(C::ConvolutionOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = ∇conv_data(v, C.stencil, C.cdims)

is_init(C::ConvolutionOperator) = ~C.cdims_onthefly && ~isnothing(C.cdims)
function initialize!(C::ConvolutionOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    C.stencil = convert(typeof(u), C.stencil)
    C.cdims = DenseConvDims(size(u), size(C.stencil); stride=C.stride, padding=C.padding, dilation=C.dilation, flipkernel=C.flipped, groups=C.groups)
end

function full_matrix(C::ConvolutionOperator{T,N}) where {T,N}

    # Apply restriction
    ((C.stride != 1) || (C.dilation != 1) || (C.groups != 1)) && trow(ArgumentError("Method not implemented for stride/dilation/groups not equal to 1")) 

    # Input check
    input_size  = domain_size(C)[1:end-1]
    output_size = range_size(C)[1:end-1]
    isnothing(input_size)  && throw(ArgumentError("The input size must be specified"))
    isnothing(output_size) && throw(ArgumentError("The output size must be specified"))

    # Size
    n_in, chs_in  = input_size[1:end-1],  input_size[end]
    _,    chs_out = output_size[1:end-1], output_size[end]

    # Zero padding matrix
    D = N-2
    P = C.padding isa Integer ? zero_padding_operator(T, D, C.padding) : zero_padding_operator(T, C.padding)
    Pm = full_matrix(P, n_in)
    padded_size = extended_size(n_in, C.padding)

    # Setting diagonal positions
    ~C.flipped ? (w = Array(reverse(C.stencil; dims=Tuple(1:D)))) : (w = Array(C.stencil))
    ssize = size(w)[1:D]
    scenter = Tuple(ones(Integer, D))
    ncprod = cumprod((1, padded_size[1:end-1]...))
    diag_pos = similar(w, Int, ssize)
    for cidx = CartesianIndices(ssize)
        diag_pos[cidx] = sum((Tuple(cidx).-scenter).*ncprod)
    end

    # Valid-convolution restriction
    padding_conv = Tuple([(j == 1) ? 0 : ssize[i]-1 for j = 1:2, i = 1:D])
    Qm = full_matrix(zero_padding_operator(T, padding_conv), reduced_size(padded_size, padding_conv))

    # Assembly sparse matrix
    m = prod(padded_size)
    return hvcat(chs_in, [Qm'*spdiagm(m, m, [diag_pos[cidx] => w[cidx, cin, cout]*ones(T, m-abs(diag_pos[cidx]))
                                             for cidx = CartesianIndices(ssize)]...)*Pm
                          for cin = 1:chs_in, cout = 1:chs_out]...)

end