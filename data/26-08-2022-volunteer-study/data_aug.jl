n1, n2 = size(corrupted)[[permutation_dims[1:2]...]]

I = CartesianIndices((n1, n2))
idx = []
for i = idx_phase_encoding
    idx_center = (I[i][1], I[i][2])
    push!(idx, idx_center)
    for index_neighboor = [(idx_center[1]-1, idx_center[2]), (idx_center[1]+1, idx_center[2]), (idx_center[1], idx_center[2]-1), (idx_center[1], idx_center[2]+1)]
        (index_neighboor[1] >= 1) && (index_neighboor[1] <= n1) && (index_neighboor[2] >= 1) && (index_neighboor[2] <= n2) && push!(idx, index_neighboor)
    end
end

L = LinearIndices((n1,n2))
idx_phase_encoding_ = []
for i = idx
    push!(idx_phase_encoding_, L[i...])
end
idx_phase_encoding_ = unique(vec(idx_phase_encoding_))
