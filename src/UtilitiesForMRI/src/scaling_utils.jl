# Resizing/Downscaling utilities

export resample, subsample, noringing_filter_1d, noringing_filter


## Spatial geometry

"""
    resample(X::CartesianSpatialGeometry, n::NTuple{3,Integer})

Up/down-sampling of Cartesian discretization geometry. The field of view and origin of `X` are kept the same, while the sampling is changed according to `n`.
"""
resample(X::CartesianSpatialGeometry, n::NTuple{3,Integer}) = spatial_geometry(X.field_of_view, n; origin=X.origin)


## k-space geometry

"""
    subsample(K, k_max::Union{T,NTuple{3,T}}; radial::Bool=false)

Down-sampling of a ``k``-space trajectory. The output is a subset of the original ``k``-space trajectory such that the ``k``-space coordinates ``(k_1,k_2,k_3)`` are:
    - ``k_i\\le k_{\\mathrm{max},i}`` (if `radial=false`), or
    - ``||\\mathbf{k}||\\le k_{\\mathrm{max}}`` (if `radial=true`).
The ordering of the subsampled trajectory is inherited from the original trajectory.
"""
function subsample(K::AbstractStructuredKSpaceSampling{T}, k_max::Union{T,NTuple{3,T}}; radial::Bool=false, also_readout::Bool=true) where {T<:Real}

    # Check input
    (k_max isa T) && (k_max = (k_max, k_max, k_max))

    # k-space coordinates & limits
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_pe1_max, k_pe2_max, k_r_max = k_max[dims_permutation(K)]

    # Scaling
    if radial
        abs_k_pe = sqrt.(dropdims(sum(k_pe.^2; dims=2); dims=2))
        pe_idx = findall(abs_k_pe .<= maximum((k_pe1_max, k_pe2_max)))
    else
        pe_idx = findall((k_pe[:,1] .< k_pe1_max) .&& (k_pe[:,1] .>= -k_pe1_max) .&& (k_pe[:,2] .< k_pe2_max) .&& (k_pe[:,2] .>= -k_pe2_max))
    end
    also_readout ? (r_idx  = findall((k_r .< k_r_max) .&& (k_r .>= -k_r_max))) : (r_idx = Colon())

    return K[pe_idx, r_idx]

end

"""
    subsample(K, X::CartesianSpatialGeometry;
              radial::Bool=false, also_readout::Bool=true)

Down-sampling of a ``k``-space trajectory, similarly to [`subsample`](@ref subsample(K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, k_max::Union{T,NTuple{3,T}}; radial::Bool=false, also_readout::Bool=true) where {T<:Real}). The maximum cutoff frequency is inferred from the Nyquist frequency of a Cartesian spatial geometry `X`.
"""
subsample(K::AbstractStructuredKSpaceSampling{T}, X::CartesianSpatialGeometry{T}; radial::Bool=false, also_readout::Bool=true) where {T<:Real} = subsample(K, Nyquist(X); radial=radial, also_readout=also_readout)


## Data array

"""
    subsample(K, d::AbstractArray{<:Complex,2}, Kq; norm_constant=nothing)

Down-sampling of a ``k``-space data array `d` associated to a ``k``-space trajectory `K`, e.g. `d_i=d(\\mathbf{k}_i)`. The output is a subset of the original data array, which is associated to the down-sampled ``k``-space `Kq` (obtained, for example, via [`subsample`](@ref subsample(K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, k_max::Union{T,NTuple{3,T}}; radial::Bool=false, also_readout::Bool=true) where {T<:Real})). The keyword `norm_constant` allows the rescaling of the down-sampled data.
"""
function subsample(K::StructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::SubsampledStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing, damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Check input
    ~isa_subsampling(Kq, K) && error("k-space subsampling mismatch")

    # Normalization
    isnothing(norm_constant) ? (d_subsampled = d[Kq.subindex_phase_encoding, Kq.subindex_readout]) : (d_subsampled = d[Kq.subindex_phase_encoding, Kq.subindex_readout]*norm_constant)

    # Filtering
    if ~isnothing(damping_factor)
        filter = noringing_filter(K, Kq; damping_factor=damping_factor)[Kq.subindex_phase_encoding, Kq.subindex_readout]
        d_subsampled .*= filter
    end

    return d_subsampled

end

"""
    subsample(K, d::AbstractArray{<:Complex,2}, Kq; norm_constant=nothing)

Down-sampling of a ``k``-space data array `d` associated to a ``k``-space trajectory `K`, e.g. `d_i=d(\\mathbf{k}_i)`. The output is a subset of the original data array, which is associated to the down-sampled ``k``-space `Kq` (obtained, for example, via [`subsample`](@ref subsample(K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, k_max::Union{T,NTuple{3,T}}; radial::Bool=false, also_readout::Bool=true) where {T<:Real})). The keyword `norm_constant` allows the rescaling of the down-sampled data.
"""
function subsample(K::CartesianStructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::SubsampledCartesianStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing, damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Check input
    ~isa_subsampling(Kq, K) && error("k-space subsampling mismatch")

    # Normalization
    isnothing(norm_constant) ? (d_subsampled = d[Kq.subindex_phase_encoding, Kq.subindex_readout]) : (d_subsampled = d[Kq.subindex_phase_encoding, Kq.subindex_readout]*norm_constant)

    # Filtering
    if ~isnothing(damping_factor)
        filter = noringing_filter(K, Kq; damping_factor=damping_factor)[Kq.subindex_phase_encoding, Kq.subindex_readout]
        d_subsampled .*= filter
    end

    return d_subsampled

end


# Reconstruction array

"""
    resample(u, n_scale::NTuple{3,Integer}; damping_factor=nothing)

Resampling of spatial array `u`. The underlying field of view represented by `u` is maintained, while the original sampling rate `n=size(u)` is changed to `n_scale`.
"""
function resample(u::AbstractArray{CT,3}, n_scale::NTuple{3,Integer}; damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Check sizes
    n = size(u)
    (n == n_scale) && (return u)

    # Computing FFT
    U_scale = zeros(CT, n_scale); idx_scale = Vector{UnitRange{Integer}}(undef,3)
    U       = fftshift(fft(u));   idx       = Vector{UnitRange{Integer}}(undef,3)
    @inbounds for i = 1:3
        if n[i] <= n_scale[i]
            idx[i]       = 1:n[i]
            idx_scale[i] = div(n_scale[i],2)+1-div(n[i],2):div(n_scale[i],2)+1+div(n[i],2)-(mod(n[i],2)==0)
        else
            idx[i]       = div(n[i],2)+1-div(n_scale[i],2):div(n[i],2)+1+div(n_scale[i],2)-(mod(n_scale[i],2)==0)
            idx_scale[i] = 1:n_scale[i]
        end
    end
    U_scale[idx_scale...] .= U[idx...]

    # Filtering
    if ~isnothing(damping_factor)
        filter = noringing_filter(T, n, n_scale; damping_factor=damping_factor)
        U_scale .*= filter[idx...]
    end
    
    return T(prod(n_scale)/prod(n))*ifft(ifftshift(U_scale))

end

function resample(u::AbstractArray{Bool,3}, n_scale::NTuple{3,Integer})

    n = size(u)
    (n == n_scale) && (return u)
    h = 1f0./(n.-1)
    idx_x_scale = Integer.(floor.(range(0f0, 1f0; length=n_scale[1])./h[1].+1))
    idx_y_scale = Integer.(floor.(range(0f0, 1f0; length=n_scale[2])./h[2].+1))
    idx_z_scale = Integer.(floor.(range(0f0, 1f0; length=n_scale[3])./h[3].+1))
    u_scale = similar(u, n_scale)
    @inbounds for i = 1:n_scale[1], j = 1:n_scale[2], k = 1:n_scale[3]
        u_scale[i,j,k] = any(u[idx_x_scale[i]:min(idx_x_scale[i]+1,n[1]),idx_y_scale[j]:min(idx_y_scale[j]+1,n[2]),idx_z_scale[k]:min(idx_z_scale[k]+1,n[3])])
    end
    return u_scale

end


# Filter utilities

function noringing_filter_1d(T::DataType, n::Integer, n_cutoff::Integer; damping_factor::Union{Real,Nothing}=nothing)
    (isnothing(damping_factor) || (n <= n_cutoff)) && (return ones(T,n))
    ((damping_factor < 0) || (damping_factor > 1)) && error("Damping factor must be between 0 and 1")
    x = convert.(T, (-div(n,2):div(n,2)-(mod(n,2)==0)))
    x_cutoff = convert(T,div(n_cutoff,2)-(mod(n_cutoff,2)==0))
    σ2 = -x_cutoff^2/(2*log(convert(T,damping_factor)))
    return exp.(-x.^2/(2*σ2))
end

function noringing_filter(T::DataType, n::NTuple{N,Integer}, n_cutoff::NTuple{N,Integer}; damping_factor::Union{Real,Nothing}=nothing) where N
    f = Vector{Array{T,N}}(undef,N)
    @inbounds for i = 1:N
        sz = ones(Integer,N); sz[i] = n[i]
        f[i] = Array{T,N}(undef,sz...)
        f[i][:] .= noringing_filter_1d(T, n[i], n_cutoff[i]; damping_factor=damping_factor)
    end
    return .*(f...)
end

function noringing_filter(K::AbstractStructuredKSpaceSampling{T}, Kq::AbstractStructuredKSpaceSampling{T}; damping_factor::Union{T,Nothing}=nothing) where {T<:Real}
    nt, nk = size(K)
    isnothing(damping_factor) && (return ones(T,nt,nk))
    k_pe,  k_r  = coord_phase_encoding(K),  coord_readout(K)
    absk2_pe = sum(k_pe.^2; dims=2); absk2_r = k_r.^2
    kq_pe, kq_r = coord_phase_encoding(Kq), coord_readout(Kq)
    max_abskq2_pe = maximum(sum(kq_pe.^2; dims=2)); max_abskq2_r = maximum(kq_r)^2
    σ2_pe = -max_abskq2_pe/(2*log(damping_factor)); σ2_r = -max_abskq2_r/(2*log(damping_factor))
    return reshape(exp.(-absk2_pe/(2*σ2_pe)), nt, 1).*reshape(exp.(-absk2_r/(2*σ2_r)), 1, nk)
end