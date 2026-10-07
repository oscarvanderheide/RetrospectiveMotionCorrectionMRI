using UtilitiesForMRI, LinearAlgebra, Test, Random
Random.seed!(123)

# Cartesian domain
n = (256, 256, 256)
L = (1.0, 1.5, 2.1)
X = spatial_geometry(L, n)

# Subsampling scheme
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:100]
r_subs = randperm(n[readout])[1:60]
K = coord(kspace_sampling(X, phase_encoding_dims; phase_encode_sampling=pe_subs, readout_sampling=r_subs))

# Adjoint test (linear operator)
P = phase_shift(K)
nt, nk, _ = size(K)
τ = randn(Float64, nt, 3)
Pτ = P(τ)
d = randn(ComplexF64, nt, nk)
e = randn(ComplexF64, nt, nk)
@test dot(Pτ*d, e) ≈ dot(d, Pτ'*e) rtol=1e-6

# Jacobian
Pτd, Pτ, ∂Pτd = ∂(P()*d, τ)
@test Pτd ≈ Pτ*d rtol=1e-6

# Adjoint test (Jacobian)
Δτ = randn(Float64, nt, 3); Δτ *= norm(τ)/norm(Δτ)
Δd_ = ∂Pτd*Δτ; Δd = randn(ComplexF64, size(Δd_)); Δd *= norm(Δd_)/norm(Δd)
@test real(dot(∂Pτd*Δτ, Δd)) ≈ dot(Δτ, ∂Pτd'*Δd) rtol=1e-6

# Gradient test
Δτ = randn(Float64, nt, 3); Δτ *= norm(τ)/norm(Δτ)
t = 1f-6
Pτd_p1 = P(τ+0.5*t*Δτ)*d
Pτd_m1 = P(τ-0.5*t*Δτ)*d
@test (Pτd_p1-Pτd_m1)/t ≈ ∂Pτd*Δτ rtol=1e-5