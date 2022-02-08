function plot_image_slice(u::AbstractArray{<:Real,2}; x::Union{Nothing,AbstractVector{<:Real}}=nothing, y::Union{Nothing,AbstractVector{<:Real}}=nothing, cmap::Union{Nothing,String}="gray", vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing, xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing, cbar_label::Union{Nothing,AbstractString}=nothing, title::Union{Nothing,AbstractString}=nothing, savefile::Union{Nothing,String}=nothing)

    isnothing(x) && (x = 1:size(u, 1))
    isnothing(y) && (y = 1:size(u, 2))
    f = figure(); ax = gca()
    imshow(u; cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax, extent=(x[1],x[end],y[end],y[1]))
    ~isnothing(xlabel) ? PyPlot.xlabel(xlabel) : ax.axes.xaxis.set_visible(false)
    ~isnothing(ylabel) ? PyPlot.ylabel(ylabel) : ax.axes.yaxis.set_visible(false)
    ~isnothing(cbar_label) && colorbar(label=cbar_label)
    PyPlot.title(title)
    ~isnothing(savefile) && savefig(savefile, dpi=300, transparent=false, bbox_inches="tight")

end

function plot_3D_result(u, vmin, vmax; x=1, y=1, z=1, filepath="", ext=".png")

    x isa Integer && (x = (x, ))
    for i = 1:length(x)
        slice = permutedims(abs.(u)[x[i],:,end:-1:1], (2,1))
        plot_image_slice(slice; vmin=vmin, vmax=vmax, savefile=string(filepath, "_x", x[i], ext))
    end
    y isa Integer && (y = (y, ))
    for i = 1:length(y)
        slice = permutedims(abs.(u)[:,y[i],end:-1:1], (2,1))
        plot_image_slice(slice; vmin=vmin, vmax=vmax, savefile=string(filepath, "_y", y[i], ext))
    end
    z isa Integer && (z = (z, ))
    for i = 1:length(z)
        slice = permutedims(abs.(u)[:,end:-1:1,z[i]], (2,1))
        plot_image_slice(slice; vmin=vmin, vmax=vmax, savefile=string(filepath, "_z", z[i], ext))
    end

end