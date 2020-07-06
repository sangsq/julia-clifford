include("sym_cliff.jl")


step_size = 10
N = 20
mid = 10

gate = mixed_gate(1,0, true)
state = all_plus(N)
for _ in 1:step_size 
    for i in 1:2:N
        gate(state, i, i % N + 1)
    end
    for i in 2:2:N
        gate(state, i, i % N + 1)
    end
end

rd_list = [2,5, 6]

max_rd = max(rd_list...)
mat = to_binary_matrix(state[:, union(1:max_rd, (mid+1):(mid+max_rd))])

for i in 1:length(rd_list)
    rd = rd_list[i]
    r1 = 1:(2rd)
    r2 = (2max_rd + 1):(2max_rd + 2rd)
    matA = view(mat, :, r1)
    matB = view(mat, :, r2)
    matAB = view(mat, :, union(r1, r2))
    @show binary_rank(matA) + binary_rank(matB) - binary_rank(matAB)
    @show mutual_info(state, 1:rd, (mid+1):(mid+rd))
end



# regionA_list = [3:5, 1:4]
# regionB_list = [8:10, 6:10]
# @show mutual_info(state, 3:5, 8:10)
# @show mutual_info(state, 1:4, 6:10)
# @show mutual_info_list(state, regionA_list, regionB_list)
