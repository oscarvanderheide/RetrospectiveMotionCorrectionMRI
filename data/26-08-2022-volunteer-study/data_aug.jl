n1, n2 = size(corrupted)[[permutation_dims[1:2]...]]

I = CartesianIndices((n1, n2))
idx = []
for i = idx_phase_encoding
    idx_center = I[i]
    push!(idx, idx_center)
    for index_neighboor = [(idx_center[1]-1, idx_center[2]), (idx_center[1]+1, idx_center[2]), (idx_center[1], idx_center[2]-1), (idx_center[1], idx_center[2]+1)]
        index = ()
        (I[p][1] >= 1) && (I[p][1] <= n1) && (I[p][2] >= 1) && (I[p][2] <= n2) && push!(idx, I[p])
    end
end