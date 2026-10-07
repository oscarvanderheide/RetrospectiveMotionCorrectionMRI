#: Concrete type

export LinearOperator, linear_operator

struct LinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    domain_size::NTuple{ND,Integer}
    range_size::NTuple{NR,Integer}
    matvecprod::Function
    matvecprod_adj::Function
    invmatvecprod::Union{Nothing,Function}
    invmatvecprod_adj::Union{Nothing,Function}
    label::Union{Nothing,String}
end

domain_size(L::LinearOperator) = L.domain_size
range_size(L::LinearOperator) = L.range_size
label(L::LinearOperator) = L.label
matvecprod(L::LinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = L.matvecprod(u)
matvecprod_adj(L::LinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = L.matvecprod_adj(v)
invmatvecprod(L::LinearOperator{TD,ND,TR,NR}, u::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = ~isnothing(L.invmatvecprod) && L.invmatvecprod(u)
invmatvecprod_adj(L::LinearOperator{TD,ND,TR,NR}, v::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = ~isnothing(L.invmatvecprod_adj) && L.invmatvecprod_adj(v)

linear_operator(TD::DataType, domain_size::NTuple{ND,Integer},
                TR::DataType, range_size::NTuple{NR,Integer},
                matvecprod::Function, matvecprod_adj::Function;
                invmatvecprod::Union{Nothing,Function}=nothing, invmatvecprod_adj::Union{Nothing,Function}=nothing,
                label::Union{Nothing,String}=nothing) where {ND,NR} =
            LinearOperator{TD,ND,TR,NR}(domain_size, range_size, matvecprod, matvecprod_adj, invmatvecprod, invmatvecprod_adj, label)