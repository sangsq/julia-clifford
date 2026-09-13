using Random
using LinearAlgebra
using Statistics
import Base:show, *, length, iterate, size, copy

mutable struct StabState
    xz::Array{Bool, 2} # (2n, 2n) shape, on each site 10->X, 01->Z, 11->XZ, 1 to n row stab, (n+1) to 2n row destab
end
copy(state::StabState) = StabState(copy(state.xz))
nsites(state::StabState) = div(size(state.xz,1), 2)

struct Clifford
    xz::Array{Bool, 2}
end

m4(x) = mod(x, 4)


function tproduct(states::AbstractVector{StabState})
    N = sum([nsites(s) for s in states])
    XZ = zeros(Bool, 2N, 2N)
    b = 0
    for s in states
        xz = s.xz
        n = nsites(s)
        XZ[b+1:b+n, 2b+1:2b+2n] = xz[1:n, 1:2n]
        XZ[N+b+1:N+b+n, 2b+1:2b+2n] = xz[n+1:2n, 1:2n]
        b += n
    end
    return StabState(XZ)
end

function all_up(n)
    xz = fill(false, 2n, 2n)
    for i in 1:n
        xz[i, 2i] = true
        xz[n+i, 2i-1] = true
    end
    return StabState(xz)
end


function all_plus(n)
    xz = fill(false, 2n, 2n)
    for i in 1:n
        xz[i, 2i-1] = true
        xz[n+i, 2i] = true
    end
    return StabState(xz)
end


function epr_pairs(n)
    xz = fill(false, 4n, 4n)
    for i in 1:n
        xz[2i-1, 2i-1] = xz[2i-1, 2n+2i-1] = true
        xz[2i, 2i] = xz[2i, 2n+2i] = true
        xz[2n+2i-1, 2i] = true
        xz[2n+2i, 2n+2i-1] = true
    end
    return StabState(xz)
end


function random_state(n)
    tmp = binary_random_symplectic_matrix(n)
    xz = zeros(Bool, 2n, 2n)
    xz[1:n, 1:2n]    = tmp[1:2:2n,1:2n]
    xz[n+1:2n, 1:2n] = tmp[2:2:2n,1:2n]
    return StabState(xz)
end


function random_clifford(n)
    xz = binary_random_symplectic_matrix(n)
    return Clifford(xz)
end

function random_css_clifford(n)
    xz = binary_random_sign_free_symplectic_matrix(n)
    return Clifford(xz)
end



function spin_to_binary_indices(spin_indices)
    n = length(spin_indices)
    indices = fill(0, 2 * n)
    for k in 1:n
        indices[2k-1] = 2 * spin_indices[k] - 1
        indices[2k] = 2 * spin_indices[k]
    end
    return indices
end


function clifford_action!(state, clifford, positions)
    n_act = length(positions)
    @assert size(clifford.xz, 1) == n_act * 2
    xz = state.xz
    n = nsites(state)
    indices = spin_to_binary_indices(positions)
    tmp_xz = fill(false, 2 * n_act)
    for k in 1:2n
        tmp_xz .= false
        for j in 1:2n_act
            if xz[k, indices[j]]
                for l in 1:n_act
                    tmp_xz[2l-1] ⊻= clifford.xz[j, 2l-1]
                    tmp_xz[2l] ⊻= clifford.xz[j, 2l]
                end

            end
        end
        xz[k, indices] = tmp_xz
    end
    return nothing
end


"""
measure out qubits in `sites' in the X basis
"""
function measure_out!(state::StabState, sites)
    xz = state.xz
    n = nsites(state)
    remain_rows = ones(Bool, 2n)
    remain_cols = ones(Bool, 2n)
    for i in sites
        bad_rows = Int[]
        for k in 1:2n
            if remain_rows[k] && xz[k, 2i]
                push!(bad_rows, k)
            end
        end
        @assert !isempty(bad_rows)
        the_row = bad_rows[1]

        xz[bad_rows[2:end], xz[the_row, :]] .⊻= true

        remain_rows[the_row] = false
        if the_row <= n
            remain_rows[the_row+n] = false
        else
            remain_rows[the_row-n] = false
        end
        remain_cols[2i-1] = false
        remain_cols[2i] = false
    end
    new_xz = xz[remain_rows, remain_cols]
    return StabState(new_xz)
end

function measure_out(state::StabState, sites)
    state = copy(state)
    return measure_out!(state, sites)
end

function contract_sites!(state::StabState, I, J)
    cliff = Clifford(Bool[1 0 0 1; 0 1 0 0; 0 0 0 1; 0 1 1 0])
    for (i, j) in zip(I, J)
        clifford_action!(state, cliff, [i, j])
    end
    sites = union(I, J)
    return measure_out!(state, sites)
end

function contract_sites(state::StabState, I, J)
    state = copy(state)
    return contract_sites!(state, I, J)
end

function reorder_sites(state::StabState, new_order)
    n = nsites(state)
    xz = state.xz
    new_xz = zeros(Bool, 2n, 2n)
    for (i, s) in enumerate(new_order)
        new_xz[:, 2i-1] = xz[:, 2s-1]
        new_xz[:, 2i] = xz[:, 2s]
    end
    return StabState(new_xz)
end



function ee_on_all_cuts(state::StabState)
    n = nsites(state)
    mat = state.xz[1:n, 2n:-1:1]
    c_rks = binary_all_vertical_cut_ranks(mat)[2n:-2:2]
    ees = [i==n ? 0 : (i - n + c_rks[i+1]) for i in 1:n]
    return ees
end


# function right_ee_on_all_cuts(state::AbstractArray{Bool, 2})
#     m, n = size(state)
#     n = div(n, 2)
#     mat = state
#     c_rks = binary_all_vertical_cut_ranks(mat)[2:2:2n]
#     ees = [i==0 ? (n-m) : (n - i - m + c_rks[i]) for i in 0:n]
#     return ees
# end

# function right_ee_on_all_cuts(state::StabState, rg)
#     m, n = size(state)
#     return right_ee_on_all_cuts(view(state.xz, 1:m, :), rg)
# end

# # function mi_on_all_cuts(state)
# #     return left_ee_on_all_cuts(state) + right_ee_on_all_cuts(state) .- entropy(state)
# # end


function mutual_info(state::AbstractArray{Bool, 2}, A, B)
    m, n = size(state)
    n = div(n, 2)
    C = setdiff(1:n, A, B)
    bA, bB, bC = spin_to_binary_indices(A), spin_to_binary_indices(B), spin_to_binary_indices(C)
    mat = state[1:m, cat(bA, bB, bC, dims=1)]
    eps = binary_bidirectional_gaussian!(mat)
    c = 0
    for i in 1:size(eps, 1)
        if (1 <= eps[i, 1] <= length(bA)) && (length(bA)+1 <= eps[i, 2] <= length(bA)+length(bB))
            c+=1
        end
    end
    return c
end


# function mutual_info(state::StabState, A, B)
#     m, n = size(state)
#     return mutual_info(view(state.xz, 1:m, :), A, B)
# end


# function mutual_neg(state::AbstractArray{Bool, 2}, region1, region2)
#     m, n = size(state)
#     n = div(n, 2)
#     l1, l2 = length(region1), length(region2)
#     new_idx = cat(region1, region2, setdiff(1:n, union(region1, region2)), dims=1)
#     new_idx = spin_to_binary_indices(new_idx)
#     mat = state[:, new_idx]
#     epoints = binary_bidirectional_gaussian!(mat)
#     mask = [i for i in 1:m if epoints[i, 2] <= 2(l1 + l2)]
#     mat1 = @views mat[mask, 1:2l1]
#     m = length(mask)
#     K = zeros(Bool, m, m)
#     for i in 1:m
#         for j in 1:i
#             @views K[i, j] = K[j, i] = binary_symplectic_inner(mat1[i, :], mat1[j, :])
#         end
#     end
#     return div(binary_rank(K), 2)
# end


# function mutual_neg(state::StabState, region1, region2)
#     m, n = size(state)
#     return mutual_neg(view(state.xz, 1:m, :), region1, region2)
# end

# """
# return [mutual_neg(state, 1:a, a+1:i) for i in a+1:n]
# """
# function multiple_negs(state::AbstractArray{Bool, 2}, a)
#     m, n = size(state)
#     n = div(n, 2)
#     mat = copy(state)

#     eps = binary_bidirectional_gaussian!(mat)
#     tmp = [(i, eps[i, 2]) for i in 1:m]
#     sort!(tmp, by= x->x[2])
#     new_order = Int[a[1] for a in tmp]
#     eps2 = eps[new_order, :]
#     right_eps = [div(eps2[i,2]+1, 2) for i in 1:m]
#     mat = mat[new_order, :]

#     gks = zeros(Int, n)
#     for i in 1:n
#         tmp = findlast(x -> x<=i, right_eps)
#         gks[i] = tmp===nothing ? 0 : tmp
#     end

#     K = zeros(Bool, m, m)
#     for i in 1:m, j in i:m
#         @views K[i, j] = K[j, i] = binary_symplectic_inner(mat[i, 1:2a], mat[j, 1:2a])
#     end
#     rank_K = binary_all_diagonal_ranks(K)

#     tmp = [gks[i]==0 ? 0 : rank_K[gks[i]] for i in a+1:n]
#     return div.(tmp, 2)
# end

# function multiple_negs(state::StabState, a)
#     m, n = size(state)
#     return multiple_negs(view(state.xz, 1:m, :), a)
# end


# function ap_negativity(state, a, b, l)
#     @assert (a<b) && (a+l<=b)
#     m, n = size(state)

#     range_list = [1:2a, 2a+2l+1:2b, 2b+2l+1:2n]
#     mat = fill(false, m, 2n)

#     @views for i in 1:l
#         mat[:, 2(2i-1)-1] = state.xz[1:m, 2(a+i)-1]
#         mat[:, 2(2i-1)] = state.xz[1:m, 2(a+i)]

#         mat[:, 2(2i)-1] = state.xz[1:m, 2(b+i)-1]
#         mat[:, 2(2i)] = state.xz[1:m, 2(b+i)]
#     end

#     i = 4l
#     @views for rg in range_list
#         len = length(rg)
#         if len>0
#             mat[:, i+1:i+len] = state.xz[1:m, rg]
#         end
#         i += len
#     end

#     end_points = binary_bidirectional_gaussian!(mat)

#     tmp = [(i, end_points[i, 2]) for i in 1:m if end_points[i, 2] <= 4l]
#     sort!(tmp, by= x->x[2])
#     new_order = Int[a[1] for a in tmp]
#     end_points2 = end_points[new_order, :]
#     m = length(new_order)

#     mask_A = [0<i%4<3 for i in 1:4l]
#     mat_A = mat[new_order, 1:4l][:, mask_A]

#     gk = fill(m, 2l)
#     j = 1
#     for i in 1:m
#         k = end_points2[i, 2]
#         spin_k = div(k+1, 2)
#         gk[j:spin_k-1] .= i-1
#         j = spin_k
#     end

#     K = zeros(Bool, m, m)
#     for i in 1:m
#         for j in 1:i
#             @views K[i, j] = K[j, i] = binary_symplectic_inner(mat_A[i, :], mat_A[j, :])
#         end
#     end
#     rank_K = binary_all_diagonal_ranks(K)
#     ngs = [gk[2r]==0 ? 0 : div(rank_K[gk[2r]], 2) for r in 1:l]
#     return ngs
# end


# @views function ap_mutual_info(state, a, b, l)
#     m, n = size(state)
#     mat = fill(false, m, 4l)

#     for i in 1:l
#         mat[:, 2(2i-1)-1] = state.xz[1:m, 2(a+i)-1]
#         mat[:, 2(2i-1)] = state.xz[1:m, 2(a+i)]

#         mat[:, 2(2i)-1] = state.xz[1:m, 2(b+i)-1]
#         mat[:, 2(2i)] = state.xz[1:m, 2(b+i)]
#     end

#     mat_AB = mat
#     mask = Bool[0<i%4<3 for i in 1:4l]
#     mat_A = mat_AB[:, mask]
#     mat_B = mat_AB[:, .!mask]

#     rk_AB = binary_all_vertical_cut_ranks!(mat_AB)[2:2:end]
#     rk_A = binary_all_vertical_cut_ranks!(mat_A)[2:2:end]
#     rk_B = binary_all_vertical_cut_ranks!(mat_B)[2:2:end]

#     mis = [rk_A[i] + rk_B[i] - rk_AB[2i] for i in 1:l]
    
#     return mis
# end


# function strange_AB_mi(state, A)
#     m, n = size(state)
#     B = div(n - A, 2)
#     mat = zeros(Bool, m, 2n)
#     mat[1:m, 1:2A] = state.xz[1:m, 2B+1:2B+2A]
#     for i in 1:B
#         mat[1:m, 2A+4i-3] = state.xz[1:m, 2(B-i+1)-1]
#         mat[1:m, 2A+4i-2] = state.xz[1:m, 2(B-i+1)]
#         mat[1:m, 2A+4i-1] = state.xz[1:m, 2(A+B+i)-1]
#         mat[1:m, 2A+4i-0] = state.xz[1:m, 2(A+B+i)]
#     end
#     ep = binary_bidirectional_gaussian!(mat)
#     mis = zeros(Int, B)
#     for i in 1:min(2A, m)
#         l = div(ep[i, 1]+1, 2)
#         r = div(ep[i, 2]+1, 2)
#         if (l <= A) && (r > A)
#             mis[div(r-A+1, 2):end] .+= 1
#         end
#     end
#     return mis
# end


# function localizable_EE(state, A, B)
#     # MI between A,B after measureing all other qubits in the X basis
#     m, n = size(state)
#     E = setdiff(1:n, A, B)
#     mat = state.xz[1:m, :]
#     is_piv_row, pivs = binary_partial_gaussian!(mat, [2i for i in E])
#     bA = spin_to_binary_indices(A)
#     bB = spin_to_binary_indices(B)
#     matA = mat[.!is_piv_row, bA]
#     matB = mat[.!is_piv_row, bB]
#     matAB = cat(matA, matB, dims=2)
#     rkA = binary_rank(matA)
#     rkB = binary_rank(matB)
#     rkAB = binary_rank(matAB)
#     return rkA + rkB - rkAB
# end


# function tri_mi(state, A, B, add_mi_AB)
#     m, n = size(state)
#     C = setdiff(1:n, A, B)
#     a, b, c = length(A), length(B), length(C)
#     bA = spin_to_binary_indices(A)
#     bB = spin_to_binary_indices(B)
#     bC = spin_to_binary_indices(C)
#     mat = zeros(Bool, m, 2n)
#     mat[1:m, 1:2a] = @view state.xz[1:m, bA]
#     mat[1:m, 2a+1:2a+2c] = @view state.xz[1:m, bC]
#     mat[1:m, 2a+2c+1:2a+2c+2b] = @view state.xz[1:m, bB]
#     ep = binary_bidirectional_gaussian!(mat)
#     mA, mB, mC, mAC, mCB = 0, 0, 0, 0, 0
#     @inline inA(x) = 0 < x <= 2a
#     @inline inC(x) = 2a < x <= 2a+2c
#     @inline inB(x) = 2a+2c < x <= 2a+2b+2c
#     for i in 1:m
#         l, r = ep[i,:]
#         if inA(r)
#             mA += 1
#             mAC += 1
#         elseif inC(l) && inC(r)
#             mC += 1
#             mAC += 1
#             mCB += 1
#         elseif inB(l)
#             mB += 1
#             mCB += 1
#         elseif inA(l) && inC(r)
#             mAC+= 1
#         elseif inC(l) && inB(r)
#             mCB+= 1
#         end
#     end
#     tri = mCB + mAC - mC - m
#     if add_mi_AB
#         tri += (a+b-entropy(state, union(A, B))) - mA - mB
#     end
#     return tri
# end