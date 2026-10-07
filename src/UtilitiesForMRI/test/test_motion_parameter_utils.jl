using UtilitiesForMRI, LinearAlgebra, Test, Random
Random.seed!(123)

nt = 256
t = Float64.(1:nt)
D = derivative1d_motionpars_linop(t, 2; pars=(true,false,true,false,true,true))
θ = zeros(Float64, nt, 6); θ[129-30:129+30,:] .= 1; θ[1:10,1] .= 1; θ[end-9:end,1] .= 1
Dθ = reshape(D*vec(θ), :, 6)

nt = 256+1
t = Float64.(0:nt-1)
ti = Float64.(0:2:nt-1); nti = length(ti)
Itp = interpolation1d_motionpars_linop((ti,ti,t,ti,ti,t), t)
θ = randn(Float64, 4*nti+2*nt)
Iθ = reshape(Itp*θ, nt, 6)

nt = 32
t = range(0.0, 1.0; length=nt)
nti = 256
ti = range(0.0, 1.0; length=nti)
Itp = interpolation1d_motionpars_linop(t, ti; interp=:nearest)
θ = randn(Float64, nt, 6)
Iθ = reshape(Itp*vec(θ), nti, 6)
Itp = interpolation1d_motionpars_linop(t, ti; interp=:spline, degree=3, tol=1e-4)
θ = randn(Float64, nt, 6)
Iθ = reshape(Itp*vec(θ), nti, 6)

# Extrapolation linear operator
fov = (1f0, 2f0, 3f0)
n = (256, 256, 256)
X = spatial_geometry(fov, n)
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:6000]
σ = 2f0  
kernel_size = 10
I = extrapolate_motionpars_linop(n[[phase_encoding_dims...]], pe_subs, nothing; T=Float32, kernel_size=kernel_size, dist_fcn=r2->exp.(-r2/(2*σ^2)))
d = randn(Float32, length(pe_subs))
d_ = reshape(I*d, n[[phase_encoding_dims...]])
# using PythonPlot
# imshow(d_)

# Consistency w/ all-parameter extrapolation operator
I6 = extrapolate_motionpars_linop(n[[phase_encoding_dims...]], pe_subs, nothing; T=Float32, kernel_size=kernel_size, dist_fcn=r2->exp.(-r2/(2*σ^2)), all_pars=true)
@test vec(repeat(I*d; outer=(1,6))) ≈ I6*vec(repeat(d; outer=(1,6))) rtol=1f-6

# Utilities for linear filling patterns
idx_local = [1,4,5,6,9,10,11,12,13]
nt = 17
θ_local = reshape(Float32.([2,2,2,3,3,3,3,4,4]), :, 1)
θ = fill_gaps(idx_local, θ_local, nt; average=true, extrapolate=true)