using BenchmarkTools
include("cliff5.jl")

function ap_negativity(state, a, b, l)
    @assert (a<b) && (a+l<=b)
    m, n = size(state)

    range_list = [1:a, a+l+1:b, b+l+1:n]
    tmp = fill(II, m, n)
    @views tmp[:, 1:2:2l] = state[:, a+1:a+l]
    @views tmp[:, 2:2:2l] = state[:, b+1:b+l]
    i = 2l
    for rg in range_list
        len = length(rg)
        if len>0
            @views tmp[:, i+1:i+len] = state[:, rg]
        end
        i += len
    end

    mat = to_binary_matrix(tmp)
    end_points = binary_bidirectional_gaussian!(mat)

    tmp2 = [(i, end_points[i, 2]) for i in 1:m if end_points[i, 2] <= 4l]
    f(x) = x[2]
    tmp2 = sort(tmp2; by=f)
    new_order = [a[1] for a in tmp2]
    mat = mat[new_order, 1:4l]
    end_points2 = end_points[new_order, :]
    m = length(new_order)

    mask_A = [0<i%4<3 for i in 1:4l]
    mat_A = mat[:, mask_A]

    gk = fill(m, 2l)
    j = 1
    for i in 1:m
        k = end_points2[i, 2]
        spin_k = div(k+1, 2)
        gk[j:spin_k-1] .= i-1
        j = spin_k
    end

    K = zeros(Bool, m, m)
    for i in 1:m
        for j in 1:i
            @views K[i, j] = K[j, i] = binary_symplectic_inner(mat_A[i, :], mat_A[j, :])
        end
    end
    rank_K = binary_all_diagonal_ranks(K)
    ngs = [gk[2r]==0 ? 0 : rank_K[gk[2r]] for r in 1:l]
    return ngs
end

# Random.seed!(1)
let
    for _ in 1:10000
        n = 10
        m = 3
        a = 1
        b = 6
        l = 4

        mat = binary_random_symplectic_matrix(n)
        state = to_stablizer_state(mat[2:2:2m, :])
        result = ap_negativity(state, a, b, l)

        r = [mutual_neg(state, a+1:a+i, b+1:b+i) for i in 1:l]
        @assert r == result
    end
end
# @show r-result

