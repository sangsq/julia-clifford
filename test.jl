include("sym_cliff.jl")

function mutual_info_list(state, regionA_list, regionB_list)
    regions = zip(regionA_list, regionB_list)
    result = Int[]
    mat = to_binary_matrix(state)
    end_points = binary_bidirectional_gaussian!(mat)
    for k in regions
        r1, r2 = k
        l1 = [a∈r1 for a in end_points[:, 1]]
        l2 = [a∈r2 for a in end_points[:, 2]]
        push!(result, sum(l1 .& l2))
    end
    return result
end


step_size = 10
N = 10

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


regionA_list = [3:5, 1:4]
regionB_list = [8:10, 6:10]
@show mutual_info(state, 3:5, 8:10)
@show mutual_info(state, 1:4, 6:10)
@show mutual_info_list(state, regionA_list, regionB_list)
