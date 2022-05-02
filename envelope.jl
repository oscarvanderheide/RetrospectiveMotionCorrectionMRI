function envelope(f::Vector{T}; window::Integer=1) where {T<:Real}
    f_env = similar(f)
    nx = length(f)
    @inbounds for x = 1:nx
        f_window = f[max(1,x-div(window,2)):min(x+div(window,2),nx)]
        f_env[x] = norm(f_window, Inf)*sign(f[x])
    end
    return f_env
end

function envelope(f::Matrix{T}; window::Integer=1) where {T<:Real}
    f_env = similar(f)
    @inbounds for i = 1:size(f,2)
        f_env[:,i] = envelope(f[:,i]; window=window)
    end
    return f_env
end