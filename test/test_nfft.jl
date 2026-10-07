using UtilitiesForMRI, LinearAlgebra, Test, Random, AbstractLinearOperators, Random
Random.seed!(123)

# Cartesian domain
fov = (1.0, 2.0, 3.0)
n = (256, 257, 256)
o = (0.5, 1.0, 1.4)
X = spatial_geometry(fov, n; origin=o)

# k-space
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:100]
r_subs = randperm(n[readout])[1:20]
K = kspace_sampling(X, phase_encoding_dims; phase_encode_sampling=pe_subs, readout_sampling=r_subs)

# Fourier operator
tol = 1e-6
norm_constant = prod(spacing(X))
F = nfft_linop(X, K; tol=tol, norm_constant=norm_constant)

# Adjoint test (linear operator)
nt, nk = size(K)
u = randn(ComplexF64, n)
d = randn(ComplexF64, nt, nk)
@test dot(F*u, d) ≈ dot(u, F'*d) rtol=1e-6

# Evaluation with rigid body motion
θ = [reshape([spacing(X)...],1,3).*randn(Float64,nt,3) 1e-2*pi*randn(Float64, nt, 3)]
Fθ = F(θ)
u = randn(ComplexF64, n)
d = randn(ComplexF64, nt, nk)
@test dot(Fθ*u, d) ≈ dot(u, Fθ'*d) rtol=1e-6

# Consistency check (with null rigid-body motion)
F_ = F(0*θ)
@test F*u ≈ F_*u rtol=1e-6
@test F'*d ≈ F_'*d rtol=1e-6

# Consistency check (parameteric operator)
F_parameteric = F()
@test F_parameteric(θ)*u  ≈ F(θ)*u  rtol=1e-6
@test F_parameteric(θ)'*d ≈ F(θ)'*d rtol=1e-6

# Consistency check (delayed eval)
@test (F()*u)(θ) ≈ F_parameteric(θ)*u rtol=1e-6

# Jacobian NFFT & consistency test
d, _, ∂Fθu = ∂(F()*u, θ)
@test d ≈ Fθ*u rtol=1e-6

# Adjoint test (Jacobian)
Δθ = randn(Float64, nt, 6); Δθ *= norm(θ)/norm(Δθ)
ΔF_ = ∂Fθu*Δθ; ΔF = randn(ComplexF64, size(ΔF_)); ΔF *= norm(ΔF_)/norm(ΔF)
@test real(dot(∂Fθu*Δθ, ΔF)) ≈ dot(Δθ, ∂Fθu'*ΔF) rtol=1e-6

# Gradient test
Δθ = randn(Float64, nt, 6); Δθ *= norm(θ)/norm(Δθ)
t = 1e-6
Fθu_p1 = F(θ+0.5*t*Δθ)*u
Fθu_m1 = F(θ-0.5*t*Δθ)*u
@test (Fθu_p1-Fθu_m1)/t ≈ ∂Fθu*Δθ rtol=1e-3

# Gauss-Newton Hessian
w = randn(ComplexF64, size(ΔF))
W = linear_operator(ComplexF64, size(ΔF), size(ΔF), d->w.*d, d-> conj(w).*d)
h = randn(ComplexF64, size(ΔF))
H = linear_operator(ComplexF64, size(ΔF), size(ΔF), d->h.*d, d-> conj(h).*d)
HF = sparse_matrix_GaussNewton(∂Fθu; W=W, H=H)

# Consistency test
Δθ = randn(Float64, nt, 6)
@test HF*vec(Δθ) ≈ vec(real(∂Fθu'*W'*(H*(W*(∂Fθu*Δθ))))) rtol=1e-6