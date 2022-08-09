using MotionCorrectedMRI, LinearAlgebra

n = (256, 256, 256)
u = zeros(Float32, n)
u[div(256,2)+1-50:div(256,2)+1+50,div(256,2)+1-50:div(256,2)+1+50,div(256,2)+1-50:div(256,2)+1+50] .= 1

p = 40f0 # dB
σ = norm(u, Inf)*10^(-p/20)
noise = σ*randn(Float32,size(u))
psnr(u+noise,u)
ssim(u+noise,u)