using UtilitiesForMRI, LinearAlgebra

# Setting noisy data
n = (256, 256, 256)
u = zeros(Float32, n)
u[div(256,2)+1-50:div(256,2)+1+50,div(256,2)+1-50:div(256,2)+1+50,div(256,2)+1-50:div(256,2)+1+50] .= 1
p = 40f0 # dB
σ = norm(u, Inf)*10^(-p/20)
noise = σ*randn(Float32,size(u))
u_noisy = u+noise

# 2D metrics
orientation = Orientation((2,1,3), (true,false,true))
nx, ny, nz = size(u)[[invperm(orientation.perm)...]]
slices = (VolumeSlice(1, div(nx,2)+1, nothing),
          VolumeSlice(2, div(ny,2)+1, nothing),
          VolumeSlice(3, div(nz,2)+1, nothing))
psnr(u_noisy, u; slices=slices, orientation=orientation)
ssim(u_noisy, u; slices=slices, orientation=orientation)

# 3D metrics
psnr(u_noisy, u)
ssim(u_noisy, u)