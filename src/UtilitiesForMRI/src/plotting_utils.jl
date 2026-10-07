export Orientation, orientation, VolumeSlice, volume_slice, FullVolume, full_volume, select, plot_volume_slice, plot_volume_slices, plot_parameters, standard_orientation


# Orientation type (custom to standard)

struct Orientation
    perm::NTuple{3,Integer}
    reverse::NTuple{3,Bool}
end

"""
    orientation(perm::NTuple{3,Integer}; reverse::NTuple{3,Bool}=(false,false,false))

Defines an orientation for a 3D image. It is mostly useful for standardizing plotting utilities. For example it can be used to reorder a 3D image such that in radiological terms, after reordering:
    - 1st dimension = left-right
    - 2nd dimension = posterior-anterior
    - 3rd dimension = inferior-superior

The input `perm` determines an ordering for the 3D dimensions, while the keyword `reverse` specifies which dimensions are negatively oriented.
"""
orientation(perm::NTuple{3,Integer}; reverse::NTuple{3,Bool}=(false,false,false)) = Orientation(perm, reverse)

"""
    standard_orientation()

Returns the standard orientation (no reordering).
"""
standard_orientation() = Orientation((1,2,3), (false,false,false))


# Volume slice type

struct FullVolume end

"""
    full_volume()

Specifies visualization options for a full 3D view.
"""
full_volume() = FullVolume()

struct VolumeSlice
    dim::Integer
    n::Integer
    window::Union{Nothing,NTuple{2,UnitRange{<:Integer}}}
end

"""
    volume_slice(dim::Integer, n::Integer; window=nothing)

Specifies a 2D slice with respect to a 3D image: `dim` is the dimension orthogonal to the 2D slice, `n` is the position of the slice, and `window` indicates a portion of the 2D slice.

Important: volume slices are always defined with respect to a given orientation of the 3D object.
"""
volume_slice(dim, n; window=nothing) = VolumeSlice(dim, n, window)

"""
    select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Orientation=standard_orientation()) where {T<:Real}

Returns the specified 2D slice of a 3D image (according to a specified orientation)
"""
function select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Orientation=standard_orientation()) where {T<:Real}
    n = size(u)
    perm_inv = invperm(orientation.perm)
    dim = perm_inv[slice.dim]
    sl = Array{Any,1}(undef,3)
    for i = 1:3
        if i == dim
            orientation.reverse[i] ? (slice_n = n[i]-slice.n+1) : (slice_n = slice.n)
            sl[i] = slice_n:slice_n
        else
            orientation.reverse[i] ? (sl[i] = n[i]:-1:1) : (sl[i] = 1:n[i])
        end
    end
    u_slice = dropdims(permutedims(u[sl...], orientation.perm); dims=slice.dim)
    isnothing(slice.window) ? (return u_slice) : (return u_slice[slice.window[1], slice.window[2]])
end

function dims(slice::VolumeSlice)
    slice.dim == 1 && (return (2,3))
    slice.dim == 2 && (return (1,3))
    slice.dim == 3 && (return (1,2))
end


# Plot utils

function plot_volume_slice(u::AbstractArray{T,2};
    extent::Union{Nothing,NTuple{4,<:Real}}=nothing,
    cmap::String="gray",
    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
    cbar_label::Union{Nothing,AbstractString}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    savefile::Union{Nothing,String}=nothing) where {T<:Real}

    figure(); ax = gca()
    imshow(u[:,end:-1:1]'; extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ~isnothing(xlabel) ? PythonPlot.xlabel(xlabel) : ax.axes.xaxis.set_visible(false)
    ~isnothing(ylabel) ? PythonPlot.ylabel(ylabel) : ax.axes.yaxis.set_visible(false)
    ~isnothing(cbar_label) && colorbar(label=cbar_label)
    PythonPlot.title(title)
    ~isnothing(savefile) && savefig(savefile, dpi=300, transparent=false, bbox_inches="tight")

end


"""
    plot_volume_slices(u::AbstractArray{T,3};
                       slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,
                       spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,
                       cmap::String="gray",
                       vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
                       xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
                       cbar_label::Union{Nothing,AbstractString}=nothing,
                       title::Union{Nothing,AbstractString}=nothing,
                       savefile::Union{Nothing,String}=nothing,
                       orientation::Orientation=standard_orientation()) where {T<:Real,N}

Plot 2D slices of a given 3D image.
"""
function plot_volume_slices(u::AbstractArray{T,3};
    slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,
    spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,
    cmap::String="gray",
    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
    cbar_label::Union{Nothing,AbstractString}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    savefile::Union{Nothing,String}=nothing,
    orientation::Orientation=standard_orientation()) where {T<:Real,N}

    if isnothing(slices)
        nx, ny, nz = size(u)[[invperm(orientation.perm)...]]
        slices = (VolumeSlice(1, div(nx,2)+1, nothing), VolumeSlice(2, div(ny,2)+1, nothing), VolumeSlice(3, div(nz,2)+1, nothing))
    end

    for n = 1:length(slices)
        isnothing(savefile) ? (savefile_slice=nothing) : (savefile_slice = string(savefile[1:end-4], "_slice", n, savefile[end-3:end]))
        slice = slices[n]
        u_slice = select(u, slice; orientation=orientation)
        x, y = coord(spatial_geometry; mesh=false)[[invperm(orientation.perm)...]][[dims(slice)...]]
        if isnothing(slice.window)
            extent = (x[1], x[end], y[end], y[1])
        else
            extent = (x[slice.window[1][1]], x[slice.window[1][end]], y[slice.window[2][end]], y[slice.window[2][1]])
        end
        plot_volume_slice(u_slice; extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label, title=title, savefile=savefile_slice)
    end

end

function plot_parameters(t::AbstractVector,
                         θ::AbstractArray, θ_ref::Union{Nothing,AbstractArray};
                         plot_flag::AbstractVector{Bool}=[true,true,true,true,true,true],
                         vmin::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing],
                         vmax::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing],
                         fmt1::Union{Nothing,AbstractString}=nothing, fmt2::Union{Nothing,AbstractString}=nothing,
                         linewidth1=2, linewidth2=1,
                         xlabel::Union{Nothing,AbstractString}="t", ylabel::Union{Nothing,AbstractVector}=[L"$\tau_x$ (mm)", L"$\tau_y$ (mm)", L"$\tau_z$ (mm)", L"$\theta_{xy}$ ($^{\circ}$)", L"$\theta_{xz}$ ($^{\circ}$)", L"$\theta_{yz}$ ($^{\circ}$)"],
                         title::Union{Nothing,AbstractString}=nothing,
                         savefile::Union{Nothing,String}=nothing,
                         orientation::Orientation=standard_orientation())

    nplots = count(plot_flag)
    _, ax = subplots(nplots, 1)
    c = 1
    isnothing(fmt1) && (fmt1 = "")
    isnothing(fmt2) && (fmt2 = "")

    # Reordering for correct orientation
    perm, sign = permutation(orientation)
    θ = deepcopy((θ.*reshape([sign...], 1, 6))[:, [perm...]])
    vmin = deepcopy(vmin)
    vmax = deepcopy(vmax)
    @inbounds for i = 1:6
        vmin_i = vmin[i]
        vmax_i = vmax[i]
        if (sign[i] == -1)
            isnothing(vmax_i) ? (vmin[i] = nothing) : (vmin[i] = -vmax_i)
            isnothing(vmin_i) ? (vmax[i] = nothing) : (vmax[i] = -vmin_i)
        end
    end
    vmin = vmin[[perm...]]
    vmax = vmax[[perm...]]

    @inbounds for i = 1:6
        if plot_flag[i]
            (i >= 4) ? (C = convert(eltype(θ), 180/pi)) : (C = 1)
            ax[c].plot(t, C*θ[:,i], fmt1, linewidth=linewidth1, label="Estimated")
            ~isnothing(θ_ref) && ax[c].plot(t, C*θ_ref[:,i], fmt2, linewidth=linewidth2, label="Reference")
            (c == 1) && ax[c].legend(loc="upper right")
            ~isnothing(xlabel) && (i == 6) && ax[c].set(xlabel=xlabel)
            (i < 6) && ax[c].get_xaxis().set_ticks([])
            ~isnothing(ylabel[i]) && ax[c].set(ylabel=ylabel[i])
            isnothing(vmin[i]) ? (vmin_i = nothing) : (vmin_i = C*vmin[i])
            isnothing(vmax[i]) ? (vmax_i = nothing) : (vmax_i = C*vmax[i])
            ax[c].set(ylim=[vmin_i, vmax_i])

            c += 1
        end
    end
    PythonPlot.title(title)
    ~isnothing(savefile) && savefig(savefile, dpi=300, transparent=false, bbox_inches="tight")

end

function permutation_rotation(perm::NTuple{2,Integer}, reverse::NTuple{2,Bool})
    (perm == (1,2)) && (order = 4; sign =  1)
    (perm == (2,1)) && (order = 4; sign = -1)
    (perm == (1,3)) && (order = 5; sign =  1)
    (perm == (3,1)) && (order = 5; sign = -1)
    (perm == (2,3)) && (order = 6; sign =  1)
    (perm == (3,2)) && (order = 6; sign = -1)
    (reverse == (true,false) || reverse == (false,true))  && (sign *= -1)
    return order, sign
end

function permutation_rotation(perm::NTuple{3,Integer}, reverse::NTuple{3,Bool})
    order = Vector{Integer}(undef,3)
    sign = Vector{Integer}(undef,3)
    @inbounds for (i, ordering) in enumerate([[1,2], [1,3], [2,3]])
        order[i], sign[i] = permutation_rotation(perm[ordering], reverse[ordering])
    end
    return order, sign
end

function permutation(orientation::Orientation)
    order_trans = orientation.perm; sign_trans = -1 .*(orientation.reverse .== true).+1 .*(orientation.reverse .== false)
    order_rot, sign_rot = permutation_rotation(orientation.perm, orientation.reverse)
    return [order_trans..., order_rot...], [sign_trans..., sign_rot...]
end