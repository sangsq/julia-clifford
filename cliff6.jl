using Random
using Profile
using LinearAlgebra
import Base:show, *, length, iterate

include("binary_linalg.jl")

mutable struct StabState
    xz::Array{Bool, 2}
    s::Array{Int, 1}
end

struct Clifford
    xz::Array{Bool, 2}
    s::Array{Int, 1}
end

const PauliString = Tuple{Int, <:AbstractArray{Bool}}

function all_up(n)
    xz = fill(false, n, 2n)
    for i in 1:n
        xz[i, 2i] = true
    end
    s = fill(false, n)
    return StabState(xz, s)
end

function all_plus(n)
    xz = fill(false, n, 2n)
    for i in 1:n
        xz[i, 2i-1] = true
    end
    s = fill(false, n)
    return StabState(xz, s)
end

function *(x::PauliString, y::PauliString)
    sx, bx = x
    sy, by = y
    s = sx + sy
    for i in 1:2:length(bx)
        s += 2 * xor(bx[i+1] * by[i])
    end
    s = s % 4
    b = Bool[xor(bx[i], by[i]) for i in 1:length(bx)]
    return s, b
end

function row_sum(state, i, j)
    ps1 = state.s[i], view(state.xz[i, :])
    ps2 = state.s[j], view(state.xz[j, :])
    r = ps1 * ps2
    state.s[j], state.xz[j, :] = r
end

commute(x1::Bool, x2::Bool, y1::Bool, y2::Bool) = !xor(x1 * y2, x2 * y1)
commute(x, y) = !binary_symplectic_inner(x, y)

function is_herm(s, xz)
    r = false
    for i in 1:2:length(xz)
        r = xor(r, xz[i] * xz[i+1])
    end
    return xor(isodd(s), r)
end

function random_clifford(n)
    xz = binary_random_symplectic_matrix(n)
    s = fill(0, 2n)
    for i in 1:2n
        h = is_herm(0, view(xz, i, :))
        s[i] = (2 * rand(Bool) + h) % 4
    end
    return Clifford(xz, s)
end


function clifford_action!(clifford, state, positions)
    n_act = length(positions)
    @assert size(clifford.xz, 1) == n_act * 2
    m, n = size(state.xz)
    indices = fill(0, 2 * n_act)
    for i in 1:n_act
        indices[2i-1] = 2 * positions[i] - 1
        indices[2i] = 2 * positions[i]
    end
    for i in 1:m
        tmp = 0, fill(false, 2 * n_act)
        for j in indices
            if state.xz[i, j]
                s = clifford.s[j]
                xz = clifford.xz[j, :]    
                tmp = tmp * (s, xz)
            end
        end
        state.s[i] = (tmp[1] + state.s[i]) % 4
        for j in indices
            state.xz[i, j] = tmp[2][j]
        end
    end
end



# function negativity(state, sub_area)
#     return binary_rank(sign_mat(state[:, sub_area]))
# end

# rk(state) = binary_rank(to_binary_matrix(state))

# function entropy(state)
#     M, N = size(state)
#     return N - M
# end

# function bipartite_entropy(state, sub_area)
#     M, N = size(state)
#     sub_size = length(sub_area)
#     sub_area_c = setdiff(1:N, sub_area)

#     c_rk = rk(state[:, sub_area_c])
#     return sub_size - M + c_rk
# end


# function pure_state_bipartite_entropy(state, sub_area)
#     # Make use of the fact that for pure state S_A = S_B
#     M, N = size(state)
#     sub_size = length(sub_area)
#     sub_area_c = setdiff(1:N, sub_area)

#     if sub_size < div(N, 2)
#         sub_area, sub_area_c = sub_area_c, sub_area
#         sub_size = N - sub_size
#     end

#     c_rk = rk(state[:, sub_area_c])
#     ee = sub_size - M + c_rk

#     return ee
# end

# function pure_ee_across_all_cuts(state, cut_point)
#     m, n = size(state)
#     mat = to_binary_matrix(state[:, 1:cut_point])
#     ees = Array(-1:-1:-cut_point)
#     pivs, _ = binary_uppertrianglize!(mat)

#     rho_left = zeros(Int, n)
#     for b_piv in pivs
#         piv = div(b_piv + 1, 2)
#         rho_left[piv] += 1
#     end

#     current_sum = 0
#     for i in 1:cut_point
#         current_sum += rho_left[i]
#         ees[i] += current_sum
#     end

#     return ees
# end


# function two_point_correlation_square(state, i, j, connected=true, d_i=Z, d_j=Z)
#     M, N = size(state)
#     a = [commute(p, d_i) for p in state[:, i]]
#     b = [commute(p, d_j) for p in state[:, j]]
#     tmp = [true for _ in 1:M]
#     result = Int(a == b) - Int(a == b == tmp) * Int(connected)
#     return result
# end


# function two_point_mutual_info(state, i, j)
#     M_AB = view(state, :, [i, j])
#     M_A = view(state, :, [i])
#     M_B = view(state, :, [j])
#     r_A, r_B, r_AB = rk(M_A), rk(M_B), rk(M_AB)
#     return r_A + r_B - r_AB
# end


# function mutual_info(state, regionA, regionB)
#     M_AB = view(state, :, union(regionA, regionB))
#     M_A = view(state, :, regionA)
#     M_B = view(state, :, regionB)
#     r_A, r_B, r_AB = rk(M_A), rk(M_B), rk(M_AB)
#     return r_A + r_B - r_AB
# end


# function mutual_neg(state, A, B)
#     sub_state = sub_area_state(state, union(A, B))
#     mn = negativity(sub_state, 1:length(A))
#     return mn
# end




# function antipodal_negativity(state, rd_list)
#     state = copy(state)
#     n = size(state, 1)
#     mid = div(n, 2)
#     state[:, 1:2:end], state[:, 2:2:end] = state[:, 1:mid], state[:, (mid+1):end]
    
#     mat = to_binary_matrix(state)
#     end_points = binary_bidirectional_gaussian!(mat)
#     tmp = [(i, end_points[i, 2]) for i in 1:n]
#     sort!(tmp, by= x-> x[2])
#     new_order = [a[1] for a in tmp]
#     mat, end_points = mat[new_order, :], end_points[new_order, :]

#     mask_A = [0<i%4<3 for i in 1:2n]
#     mat_A = mat[:, mask_A]

#     gk = [n for _ in 1:n]
#     j = 1
#     for i in 1:n
#         k = end_points[i, 2]
#         spin_k = div(k+1, 2)
#         gk[j:spin_k-1] .= i-1
#         j = spin_k
#     end

#     K = zeros(Bool, n, n)
#     for i in 1:n
#         for j in 1:i
#             K[i, j] = K[j, i] = binary_symplectic_inner(mat_A[i, :], mat_A[j, :])
#         end
#     end
#     rank_K = binary_all_diagonal_ranks(K)
#     # ngs = [binary_rank(K[1:gk[2r], 1:gk[2r]]) for r in rd_list]
#     ngs = [gk[2r]==0 ? 0 : rank_K[gk[2r]] for r in rd_list]
#     return ngs
# end

# function antipodal_mutual_info(state, rd_list)
#     state = copy(state)
#     n = size(state, 1)
#     mid = div(n, 2)
#     state[:, 1:2:end], state[:, 2:2:end] = state[:, 1:mid], state[:, (mid+1):end]
    
#     mat_AB = to_binary_matrix(state)
#     mask = [0<i%4<3 for i in 1:2n]
#     mat_A = mat_AB[:, mask]
#     mat_B = mat_AB[:, .!mask]

#     end_points_AB = binary_bidirectional_gaussian!(mat_AB)
#     end_points_A = binary_bidirectional_gaussian!(mat_A)
#     end_points_B = binary_bidirectional_gaussian!(mat_B)

#     rk_AB = [0 for _ in 1:n]
#     j = 1  
#     for i in 1:n
#         k = end_points_AB[i, 1]
#         if k==0
#             rk_AB[j:end] .= i-1
#             break
#         end
#         spin_k = div(k+1, 2)
#         rk_AB[j:spin_k-1] .= i-1
#         j = spin_k
#         (i==n) && (rk_AB[j:end] .= n)
#     end

#     rk_A = [0 for _ in 1:div(n,2)]
#     j = 1
#     for i in 1:n
#         k = end_points_A[i, 1]
#         if k==0
#             rk_A[j:end] .= i-1
#             break
#         end
#         spin_k = div(k+1, 2)
#         rk_A[j:spin_k-1] .= i-1
#         j = spin_k
#         (i==n) && (rk_A[j:end] .= n)
#     end

#     rk_B = [0 for _ in 1:div(n,2)]
#     j = 1
#     for i in 1:n
#         k = end_points_B[i, 1]
#         if k==0
#             rk_B[j:end] .= i-1
#             break
#         end
#         spin_k = div(k+1, 2)
#         rk_B[j:spin_k-1] .= i-1
#         j = spin_k
#         (i==n) && (rk_B[j:end] .= n)
#     end

#     mis = [rk_A[i] + rk_B[i] - rk_AB[2i] for i in rd_list]
    
#     return mis
# end


# function several_mutual_info(state, a, rd_list_a, b, rd_list_b)
#     n = size(state, 1)
#     result = zeros(Int, length(rd_list_a),  length(rd_list_b))
#     bmat = to_binary_matrix(state)
#     mat_B = bmat[:, 2b+1:2b+2rd_list_b[end]]
#     rk_B = binary_all_vertical_cut_ranks!(mat_B)[2:2:end]

#     for i in 1:length(rd_list_a)
#         ra = rd_list_a[i]
#         mat_A = bmat[:, 2a+1:2a+2ra]
#         mat_AB = bmat[:, union(2a+1:2a+2ra, 2b+1:2b+2rd_list_b[end])] 
#         rk_A = binary_rank(mat_A)
#         rk_AB = binary_all_vertical_cut_ranks!(mat_AB)[2:2:end]
#         result[i, :] = [rk_A + rk_B[l] - rk_AB[ra+l] for l in rd_list_b]
#     end

#     return result
# end


