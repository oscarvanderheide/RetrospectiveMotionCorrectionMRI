using LinearAlgebra, UtilitiesForMRI, JLD, PyPlot, Images
using Random; Random.seed!(123)

# Setting folder/savefiles
phantom_id = "invivo3D"
motion_id  = "sudden"
data_folder    = string(pwd(), "/data/",    phantom_id, "/")
results_folder = string(pwd(), "/results/", phantom_id, "/"); try mkdir(results_folder) catch end
figures_folder = string(pwd(), "/figures/", phantom_id, "/"); try mkdir(figures_folder) catch end

# Loading data
u = ComplexF32.(load(string(data_folder, phantom_id, ".jld"))["T2_nomotion"])
u = u/norm(u, Inf)

# Setting rbm operators
n = size(u)
X = spatial_sampling(n; h=[1f0,1f0,1f0])
K = kspace_sampling(X; phase_encode=:xy, readout=:z)
nt, nk = size(K)
F = nfft(X, K; tol=1f-5)

# Setting random motion
nf = 2^5
θ_lowdim = randn(Float32, nf, 6)
θ = imresize(θ_lowdim, nt, 6)
for i = 1:3
    θ[:, i] .*= 0.0125f0*n[1]/norm(θ[:, i], Inf)
end
for i = 4:6
    θ[:, i] .*= 5f0*pi/(180f0*norm(θ[:, i], Inf))
end

# Utils
F0 = F(zeros(Float32, nt, 6))
function plot_fig(t)
    # clf()
    θt = repeat(θ[t:t,:]; outer=(nt, 1))
    ut = F0'*(F(θt)*u)
    subplot(1,3,1)
    imshow(permutedims(abs.(ut[129,:,:]), (2,1))[end:-1:1,:]; vmin=0, vmax=1, cmap="gray")
    subplot(1,3,2)
    imshow(permutedims(abs.(ut[:,129,:]), (2,1))[end:-1:1,:]; vmin=0, vmax=1, cmap="gray")
    subplot(1,3,3)
    imshow(abs.(ut[:,:,129]); vmin=0, vmax=1, cmap="gray")
    for i = 1:3
        plt = subplot(1,3,i); plt.axes.xaxis.set_visible(false); plt.axes.yaxis.set_visible(false)
    end
end

# Animate
t = 1:2^10:nt
for i = 1:length(t)
    println(i, "/", length(t))
    plot_fig(t[i])
    savefig(string("./figures/animation_rbm/frame_", i, ".png"), dpi=300, transparent=false, bbox_inches="tight")
end

# Reconstruction
u_recon = F0'*(F(θ)*u)
subplot(1,3,1)
imshow(permutedims(abs.(u_recon[129,:,:]), (2,1))[end:-1:1,:]; vmin=0, vmax=1, cmap="gray")
subplot(1,3,2)
imshow(permutedims(abs.(u_recon[:,129,:]), (2,1))[end:-1:1,:]; vmin=0, vmax=1, cmap="gray")
subplot(1,3,3)
imshow(abs.(u_recon[:,:,129]); vmin=0, vmax=1, cmap="gray")
for i = 1:3
    plt = subplot(1,3,i); plt.axes.xaxis.set_visible(false); plt.axes.yaxis.set_visible(false)
end
savefig(string("./figures/animation_rbm/recon.png"), dpi=300, transparent=false, bbox_inches="tight")