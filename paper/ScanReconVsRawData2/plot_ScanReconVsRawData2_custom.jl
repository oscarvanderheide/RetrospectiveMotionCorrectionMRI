using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PythonPlot, JLD

# Load data
results_folder = string(pwd(), "/paper/ScanReconVsRawData2/")
results_file = string(results_folder, "ScanReconVsRawData2_custom.jld")
corrupted, ground_truth, X, u_reg, u, θ, prior = load(results_file, "corrupted", "ground_truth", "X", "u_reg", "u", "θ", "prior")
orientation = Orientation((2,1,3), (true,false,true))

# Registration
opt_reg = rigid_registration_options(; niter=20, verbose=true)
u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)

# Denoising
h = spacing(X)
opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
ε = g(u)
corrupted_reg = proj(corrupted_reg, ε, g)
u_reg = proj(u_reg, ε, g)
ground_truth_reg = proj(ground_truth, ε, g)

# Masking
mask = abs.(ground_truth).>1f-3*norm(ground_truth,Inf)
C = zero_set(ComplexF32, (!).(mask))
u_reg = proj(u_reg, C)
corrupted_reg = proj(corrupted_reg, C)
ground_truth_reg = proj(ground_truth_reg, C)
@save string(results_folder, "post_processed_scans_custom.jld") u_reg corrupted_reg ground_truth_reg

# Reconstruction quality
nx, ny, nz = size(ground_truth)[[invperm(orientation.perm)...]]
slices = (VolumeSlice(1, div(nx,2)+1, nothing),
          VolumeSlice(2, div(ny,2)+1, nothing),
          VolumeSlice(3, div(nz,2)+1, nothing))
fact = norm(ground_truth_reg, Inf)
psnr_corrected = psnr(abs.(u_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
psnr_corrupted = psnr(abs.(corrupted_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
ssim_corrected = ssim(abs.(u_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
ssim_corrupted = ssim(abs.(corrupted_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
@save string(results_folder, "indexes_custom.jld") psnr_corrected psnr_corrupted ssim_corrected ssim_corrupted

# Plot results
plot_volume_slices(abs.(ground_truth_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth_reg,Inf), savefile=string(results_folder, "ground_truth_custom.eps"), orientation=orientation)
plot_volume_slices(abs.(prior); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference_custom.eps"), orientation=orientation)
plot_volume_slices(abs.(u_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth_reg,Inf), savefile=string(results_folder, "corrected_custom.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth_reg,Inf), savefile=string(results_folder, "corrupted_custom.eps"), orientation=orientation)

# Plot results (detail)
c = (div(nx,2)+21, div(ny,2)+31)
l = 50
slices = (VolumeSlice(3, div(nz,2)+1, (c[1]-l:c[1]+l, c[2]-l:c[2]+l)), )
plot_volume_slices(abs.(ground_truth_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "ground_truth_custom_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(prior); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(prior,Inf), savefile=string(results_folder, "reference_custom_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(u_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrected_custom_detail.eps"), orientation=orientation)
plot_volume_slices(abs.(corrupted_reg); slices=slices, spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(results_folder, "corrupted_custom_detail.eps"), orientation=orientation)

# Plot motion parameters
# θ[:, 1:3] .*= 1e3
θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
# Δθ[1:3] .= 1
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding index", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(results_folder, "parameters_motion_custom.eps"))
close("all")