using LinearAlgebra, AbstractProximableFunctions, AbstractLinearOperators, Test, Random
Random.seed!(123)

# Random input
T = Float64
n = (1024, 2048)

# AbstractLinearOperators
fw = x -> x[1:2:end,1:2:end]
function bw(y)
    x = zeros(T,n)
    x[1:2:end,1:2:end] .= y
    return x
end
A = linear_operator(T, n, (512, 1024), fw, bw)

# Misfit
y = randn(T, 512, 1024)
J = leastsquares_misfit(A, y)

# Gradient test
x = randn(T, n...)
@test test_grad(J, x; step=T(1e-5), rtol=T(1e-6))