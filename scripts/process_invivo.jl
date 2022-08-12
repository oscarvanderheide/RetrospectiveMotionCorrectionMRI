using LinearAlgebra, RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, Flux, PyPlot, JLD, Statistics

# In-vivo data
prior = load("./data/Invivo3D/Invivo3D_unprocessed.jld")["T1_nomotion"]
ground_truth = load("./data/Invivo3D/Invivo3D_unprocessed.jld")["FLAIR_nomotion"]
u_data = load("./data/Invivo3D/Invivo3D_unprocessed.jld")["FLAIR_motion"]

# Denoising prior
h = (1f0, 1f0, 1f0)
n = size(prior)
g = gradient_norm(2, 1, n, h; T=ComplexF32)
ε = 0.5f0*g(prior)
opt_proj = opt_fista(1f0/12f0; niter=20, Nesterov=true)
prox(u, _) = project(u, ε, g, opt_proj)
prior = prox(prior, nothing)

# Tapering
i0 = 40
i1 = 70
taper = zeros(Float32, 1, 1, n[3]); taper[i1+1:end] .= 1
taper[i0+1:i1] .= (sin.(range(-pi/2, pi/2; length=i1-i0)).+1)/2
prior .*= taper
ground_truth .*= taper

# Registration
function grad(u::AbstractArray{CT,3}; η::T=T(1)) where {T<:Real,CT<:Union{T,Complex{T}}}
    ∇u = similar(u, size(u).-1)
    ∇u .= abs.(u[2:end,1:end-1,1:end-1]-u[1:end-1,1:end-1,1:end-1])+
          abs.(u[1:end-1,2:end,1:end-1]-u[1:end-1,1:end-1,1:end-1])+
          abs.(u[1:end-1,1:end-1,2:end]-u[1:end-1,1:end-1,1:end-1])
    return ∇u./(∇u.+mean(∇u)*η)
end

# Optimization options
niter = 128
opt = rigid_registration_options(; niter=niter, steplength=1f0, λ=0f0, cdiag=0f0, cid=0f0, verbose=true)

# Solution
u_mov = imresize(u_T1, (32,32,32)); u_mov = grad(u_mov)
u_fix = imresize(u_T2, (32,32,32)); u_fix = grad(u_fix)
θ = nothing
_, θ, _ = rigid_registration(u_mov/norm(u_mov), u_fix/norm(u_fix), θ, opt)

u_mov = imresize(u_T1, (64,64,64)); u_mov = grad(u_mov)
u_fix = imresize(u_T2, (64,64,64)); u_fix = grad(u_fix)
opt.niter = 64
θ[:, 1:3] .*= 2
_, θ, _ = rigid_registration(u_mov/norm(u_mov), u_fix/norm(u_fix), θ, opt)

u_mov = imresize(u_T1, (128,128,128)); u_mov = grad(u_mov)
u_fix = imresize(u_T2, (128,128,128)); u_fix = grad(u_fix)
opt.niter = 16
θ[:, 1:3] .*= 2
_, θ, _ = rigid_registration(u_mov/norm(u_mov), u_fix/norm(u_fix), θ, opt)

u_mov = imresize(u_T1, (256,256,256)); u_mov = grad(u_mov)
u_fix = imresize(u_T2, (256,256,256)); u_fix = grad(u_fix)
opt.niter = 8
θ[:, 1:3] .*= 2
_, θ, _ = rigid_registration(u_mov/norm(u_mov), u_fix/norm(u_fix), θ, opt)

n = size(u_T1)
X = spatial_sampling(n; h=Float32.([1,1,1]))
K = KSpaceFixedSizeSampling{Float32}(reshape(kspace_sampling(X).K, 1, prod(n), 3))
F = nfft(X, K)
u_T1_reg = F(zeros(Float32, 1, 6))'*(F(θ)*u_T1)

T1_nomotion = u_T1_reg
T2_nomotion = load("./data/invivo3D/invivo3D_unprocessed.jld")["T2_nomotion"]
T2_motion = u_T2
@save "./data/invivo3D/invivo3D.jld" T1_nomotion T2_nomotion T2_motion

include(string(pwd(), "/scripts/plot_results.jl"))
plot_3D_result(T1_nomotion, 0, norm(T1_nomotion,Inf); x=129, y=129, z=129, filepath="T1", ext=".png")
plot_3D_result(T2_nomotion, 0, norm(T2_nomotion,Inf); x=129, y=129, z=129, filepath="T2", ext=".png")
plot_3D_result(T2_motion, 0, norm(T2_motion,Inf); x=129, y=129, z=129, filepath="T2_motion", ext=".png")