include("cliff5.jl")



function several_mutual_neg(state, a, rd_list_a, b, rd_list_b)
    m, n = size(state)
    result = zeros(Int, length(rd_list_a),  length(rd_list_b))
    @assert m==n
    for i in 1:length(rd_list_a)
        ra = rd_list_a[i]
        state_c = view(state, :, union(a+1:a+ra, b+1:b+rd_list_b[end], 1:a, a+ra+1:b, b+rd_list_b[end]:n))
        mat = to_binary_matrix(state_c)
        end_points = binary_bidirectional_gaussian!(mat)
        tmp = [(i, end_points[i, 2]) for i in 1:n]
        sort!(tmp, by= x-> x[2])
        new_order = [x[1] for x in tmp]
        mat, end_points = mat[new_order, :], end_points[new_order, :]
        mat_A = mat[:, 1:2ra]
        gk = [n for _ in 1:n]
        j = 1
        for i in 1:n
            k = end_points[i, 2]
            spin_k = div(k+1, 2)
            gk[j:spin_k-1] .= i-1
            j = spin_k
        end
        K = zeros(Bool, n, n)
        for x in 1:n, y in 1:x
            K[x, y] = K[y, x] = binary_symplectic_inner(mat_A[x, :], mat_A[y, :])
        end
        rank_K = binary_all_diagonal_ranks(K)
        result[i, :] = [gk[ra+rb]==0 ? 0 : rank_K[gk[ra+rb]] for rb in rd_list_b]
    end
    return result
end


N = 50
a = 1
b = 25
a_list=1:20
b_list=1:20

Random.seed!(1)
mat = binary_random_symplectic_matrix(N)
state = to_stablizer_state(mat[1:N, :])

result = several_mutual_neg(state, a, a_list, b, b_list)

r = zeros(Int, length(a_list), length(b_list))

for i in 1:length(a_list), j in 1:length(b_list)
    r[i,j] = mutual_neg(state, a+1:a+a_list[i], b+1:b+b_list[j])
end

@show result - r
