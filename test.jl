include("sym_cliff.jl")
using Profile

# function bn(n)
#     tmp = zeros(Bool, n, n)
#     [tmp[i,i]=true for i in 1:n]
#     tmp = .!tmp
#     return binary_rank(tmp)
# end


# open("../prof.txt", "w") do s
#     Profile.print(IOContext(s, :displaysize => (24, 500)))
# end
function several_mutual_info1(state, a, rd_list_a, b, rd_list_b)
    n = size(state, 1)
    result = zeros(Int, length(rd_list_a),  length(rd_list_b))
    bmat = to_binary_matrix(state)
    mat_B = bmat[:, 2b+1:2b+2rd_list_b[end]]
    rk_B = binary_all_vertical_cut_ranks!(mat_B)[2:2:end]

    for i in 1:length(rd_list_a)
        ra = rd_list_a[i]
        mat_A = bmat[:, 2a+1:2a+2ra]
        mat_AB = bmat[:, union(2a+1:2a+2ra, 2b+1:2b+2rd_list_b[end])] 
        rk_A = binary_rank(mat_A)
        rk_AB = binary_all_vertical_cut_ranks!(mat_AB)[2:2:end]
        result[i, :] = [rk_A + rk_B[l] - rk_AB[ra+l] for l in rd_list_b]
        for l in rd_list_b
            @show rk_A, rk_B[l], rk_AB[ra+l]
        end
    end
    return result
end

function mutual_info1(state, regionA, regionB)
    M_AB = view(state, :, union(regionA, regionB))
    M_A = view(state, :, regionA)
    M_B = view(state, :, regionB)
    r_A, r_B, r_AB = rk(M_A), rk(M_B), rk(M_AB)
    @show r_A, r_B, r_AB
    return r_A + r_B - r_AB
end

Random.seed!(1111)

N = 100
a = 5
b = 50
step_size = 100
rd_list_a = [10, 30, 40]
rd_list_b = [9, 14, 30]
gate = mixed_gate(0.5, 0.3, false)
state = all_plus(N)
for _ in 1:step_size 
    for i in 1:2:N
        gate(state, i, i % N + 1)
    end
    for i in 2:2:N
        gate(state, i, i % N + 1)
    end
    r1 = several_mutual_info(state, a, rd_list_a, b, rd_list_b)
    r2 = zeros(Int, 3, 3)
    for i in 1:3
        for j in 1:3
            r2[i, j] = mutual_info(state, a+1:a+rd_list_a[i], b+1:b+rd_list_b[j])
        end
    end
    if r1 != r2
        break
    end
end

# r1 = several_mutual_info1(state, a, rd_list_a, b, rd_list_b)
# r2 = zeros(Int, 3, 3)
# for i in 1:3
#     for j in 1:3
#         r2[i, j] = mutual_info1(state, a+1:a+rd_list_a[i], b+1:b+rd_list_b[j])
#     end
# end
# if r1 != r2
#     @show r1
#     @show r2
# end


# @time antipodal_mutual_info(state, rd_list)
# @time antipodal_negativity(state, rd_list)