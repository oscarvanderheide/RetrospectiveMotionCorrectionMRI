# Loss function utilities

export MixedNorm, MixedNormMoreau
export Lipschitz_constant, data_residual_loss

abstract type AbstractLossFunction{T} end
abstract type AbstractLossFunctionMoreau{T}<:AbstractLossFunction{T} end


## Mixed norms

struct MixedNorm{T,N1,N2}<:AbstractLossFunction{T}
    regularized_Hessian::Bool
end

data_residual_loss(T::DataType, N1::Integer, N2::Integer; ρ::Real=Inf, regularized_Hessian::Bool=false) = isinf(ρ) ? MixedNorm{T,N1,N2}(regularized_Hessian) : MixedNormMoreau{T,N1,N2}(real(T)(ρ), regularized_Hessian)

### (2,2)

function (::MixedNorm{CT,2,2})(d::AbstractArray{CT,2}; Hessian::Bool=false) where {T<:Real,CT<:RealOrComplex{T}}
    f = T(0.5)*norm(d)^2
    g = copy(d)
    Hessian ? (return (f, g, identity_operator(CT, size(d)))) : (return (f, g))
end

Lipschitz_constant(::MixedNorm{CT,2,2}) where {T<:Real,CT<:RealOrComplex{T}} = T(1)

### (2,1)

function (ℓ::MixedNorm{CT,2,1})(d::AbstractArray{CT,2}; Hessian::Bool=false) where {T<:Real,CT<:RealOrComplex{T}}
    normd = sqrt.(sum(abs.(d).^2; dims=2))
    f = sum(normd)
    g = d./normd
    if Hessian
        ℓ.regularized_Hessian ? (evalH = d̄->d̄./normd) : (evalH = d̄->(d̄-g.*real(sum(conj(g).*d̄; dims=2)))./normd)
        H = linear_operator(CT, size(d), size(d), evalH, evalH)
    end
    Hessian ? (return (f, g, H)) : (return (f, g))
end


## Moreau norms

struct MixedNormMoreau{T,N1,N2}<:AbstractLossFunctionMoreau{T}
    ρ::Real
    regularized_Hessian::Bool
end

function (ℓ::MixedNormMoreau{CT,2,1})(d::AbstractArray{CT,2}; Hessian::Bool=false) where {T<:Real,CT<:RealOrComplex{T}}
    normd = sqrt.(sum(abs.(d).^2; dims=2))
    d_ = similar(d)
    idx_threshold = vec(normd) .> ℓ.ρ
    d_[(!).(idx_threshold), :] .= 0
    d_[idx_threshold, :] .= d[idx_threshold, :].*(T(1).-ℓ.ρ./normd[idx_threshold])
    g = (d-d_)/ℓ.ρ
    f = sum(sqrt.(sum(abs.(d_).^2; dims=2)))+T(0.5)*ℓ.ρ*norm(g)^2
    if Hessian
        if ℓ.regularized_Hessian
            hinv = similar(d, size(d, 1), 1)
            hinv[(!).(idx_threshold)] .= ℓ.ρ
            hinv[idx_threshold]       .= normd[idx_threshold]
        end
        function evalH(d̄::AbstractArray{CT,2})
            if ℓ.regularized_Hessian
                return d̄./hinv
            else
                Hd = similar(d̄)
                Hd[(!).(idx_threshold), :] .= d̄[(!).(idx_threshold), :]/ℓ.ρ
                Hd[idx_threshold, :] .= (d̄[idx_threshold, :]-d[idx_threshold, :].*real(sum(conj(d[idx_threshold, :]).*d̄[idx_threshold, :]; dims=2))./normd[idx_threshold].^2)./normd[idx_threshold]
                return Hd
            end
        end
        H = linear_operator(CT, size(d), size(d), evalH, evalH)
    end
    Hessian ? (return (f, g, H)) : (return (f, g))
end

Lipschitz_constant(ℓ::MixedNormMoreau{CT,2,1}) where {T<:Real,CT<:RealOrComplex{T}} = T(1)/ℓ.ρ