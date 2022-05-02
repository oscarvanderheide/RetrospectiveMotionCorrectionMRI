function plot_image_slice(u::AbstractArray{<:Real,2}; h::Union{Nothing,NTuple{2,<:Real}}=nothing, cmap::Union{Nothing,String}="gray", vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing, xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing, cbar_label::Union{Nothing,AbstractString}=nothing, title::Union{Nothing,AbstractString}=nothing, savefile::Union{Nothing,String}=nothing)

    isnothing(h) && (h = (1, 1))
    L = (size(u).-1).*h
    extent = (0, L[2], L[1], 0)
    f = figure(); ax = gca()
    imshow(u; aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ~isnothing(xlabel) ? PyPlot.xlabel(xlabel) : ax.axes.xaxis.set_visible(false)
    ~isnothing(ylabel) ? PyPlot.ylabel(ylabel) : ax.axes.yaxis.set_visible(false)
    ~isnothing(cbar_label) && colorbar(label=cbar_label)
    PyPlot.title(title)
    ~isnothing(savefile) && savefig(savefile, dpi=300, transparent=false, bbox_inches="tight")

end

function plot_3D_result(u, vmin, vmax; h::Union{Nothing,NTuple{3,<:Real}}=nothing, x=1, y=1, z=1, filepath="", ext=".png", aspect::Union{Nothing,String}=nothing)

    x isa Integer && (x = (x, ))
    isnothing(h) && (h = (1, 1, 1))
    for i = 1:length(x)
        slice = permutedims(abs.(u)[x[i],:,end:-1:1], (2,1))
        plot_image_slice(slice; h=(h[3], h[2]), vmin=vmin, vmax=vmax, savefile=string(filepath, "_x", x[i], ext))
    end
    y isa Integer && (y = (y, ))
    for i = 1:length(y)
        slice = permutedims(abs.(u)[:,y[i],end:-1:1], (2,1))
        plot_image_slice(slice; h=(h[3], h[1]), vmin=vmin, vmax=vmax, savefile=string(filepath, "_y", y[i], ext))
    end
    z isa Integer && (z = (z, ))
    for i = 1:length(z)
        slice = permutedims(abs.(u)[:,end:-1:1,z[i]], (2,1))
        plot_image_slice(slice; h=(h[2], h[1]), vmin=vmin, vmax=vmax, savefile=string(filepath, "_z", z[i], ext))
    end

end

function plot_parameters(t::AbstractVector, θ::AbstractArray, θ_ref::Union{Nothing,AbstractArray}; plot_flag::AbstractVector{Bool}=[true,true,true,true,true,true], vmin::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing], vmax::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing], fmt1::Union{Nothing,AbstractString}=nothing, fmt2::Union{Nothing,AbstractString}=nothing, linewidth1=2, linewidth2=1, xlabel::Union{Nothing,AbstractString}="t", ylabel::Union{Nothing,AbstractVector}=[L"$\tau_x$ (px)", L"$\tau_y$ (px)", L"$\tau_z$ (px)", L"$\theta_{xy}$ ($^{\circ}$)", L"$\theta_{xz}$ (rad)", L"$\theta_{yz}$ ($^{\circ}$)"], filepath="", ext=".png")

    nplots = count(plot_flag)
    _, ax = subplots(nplots, 1)
    # _, ax = subplots(1, nplots)
    c = 1
    for i = 1:6
        if plot_flag[i]
            (i >= 4) ? (C = 180/pi) : (C = 1)
            ax[c].plot(t, C*θ[:,i],     fmt1, linewidth=linewidth1, label="Estimated")
            ~isnothing(θ_ref) && ax[c].plot(t, C*θ_ref[:,i], fmt2, linewidth=linewidth2, label="Reference")
            (c == 1) && ax[c].legend(loc="upper right")
            # ax[i].plot(θ[:,i],     t, fmt1, linewidth=linewidth)
            # ax[i].plot(θ_ref[:,i], t, fmt2, linewidth=linewidth)
            ~isnothing(xlabel) && (i == 6) && ax[c].set(xlabel=xlabel)
            (i < 6) && ax[c].get_xaxis().set_ticks([])
            ~isnothing(ylabel[i]) && ax[c].set(ylabel=ylabel[i])
            # ~isnothing(ylabel[i]) && ax[c].set(xlabel=ylabel[i])
            # ~isnothing(xlabel[i]) && ax[c].set(ylabel=xlabel[i])
            if isnothing(vmin[i])
                ~isnothing(θ_ref) ? (vmin_i = minimum(C*θ_ref[:,i])) : (vmin_i = minimum(C*θ[:,i]))
            else
                vmin_i = vmin[i]
            end
            if isnothing(vmax[i])
                ~isnothing(θ_ref) ? (vmax_i = maximum(C*θ_ref[:,i])) : (vmax_i = maximum(C*θ[:,i]))
            else
                vmax_i = vmax[i]
            end
            Δv = vmax_i-vmin_i
            ax[c].set(ylim=[vmin_i-0.1*Δv, vmax_i+0.1*Δv])
            # ax[c].set(xlim=[vmin-0.1*Δv, vmax+0.1*Δv])

            c += 1
        end
    end
    mngr = get_current_fig_manager()
    mngr.window.setGeometry(50, 100, 700, 1000); pause(0.1)
    savefig(string(filepath, ext), dpi=300, transparent=false, bbox_inches="tight")

end