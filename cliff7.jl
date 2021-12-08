using Random
using LinearAlgebra
using Statistics
import Base:show, *, length, iterate, size, copy

include("binary_linalg.jl")

mutable struct StabState
    xz::Array{Bool, 2} # (2n, 2n) shape, on each site 10->X, 01->Z, 11->XZ, 1 to n row stab, (n+1) to 2n row destab
    s::Array{Int, 1} # shape (2n,), {0, 1, 2, 3} -> {1, i, -1, -i}
    n_stab::Int # number of stablizer for the state
end

copy(state::StabState) = StabState(copy(state.xz), copy(state.s), state.n_stab)

struct Clifford
    xz::Array{Bool, 2}
    s::Array{Int, 1}
end

flat(state::StabState) = state.xz, state.s, state.n_stab

m4(x) = mod(x, 4)

size(state::StabState) = state.n_stab, div(size(state.xz, 1), 2)
function size(state::StabState, dim::Int)
    @assert dim == 1 || dim == 2
    if dim==1
        return state.n_stab
    else
        return div(size(state.xz, 2), 2)
    end
end

# const PauliString = Tuple{Int, <:AbstractArray{Bool}}
PauliString = Tuple{Int, <:AbstractArray{Bool}}

function all_up(n)
    xz = fill(false, 2n, 2n)
    for i in 1:n
        xz[i, 2i] = true
        xz[n+i, 2i-1] = true
    end
    s = fill(0, 2n)
    n_stab = n
    return StabState(xz, s, n_stab)
end


function all_plus(n)
    xz = fill(false, 2n, 2n)
    for i in 1:n
        xz[i, 2i-1] = true
        xz[n+i, 2i] = true
    end
    s = fill(0, 2n)
    n_stab = n
    return StabState(xz, s, n_stab)
end

function white_state(n)
    xz = fill(false, 2n, 2n)
    s = fill(0, 2n)
    n_stab = 0
    return StabState(xz, s, n_stab)
end

function partial_up(n, p)
    xz = fill(false, 2n, 2n)
    s = fill(0, 2n)
    j = 0
    for i in 1:n
        if rand() < p
            j += 1
            xz[j, 2i] = true
            xz[j+n, 2i-1] = true
        end
    end
    return StabState(xz, s, j)
end

function random_state(n, n_stab)
    tmp = random_clifford(n)
    xz = zeros(Bool, 2n, 2n)
    s = zeros(Int, 2n)
    @views for i in 1:n_stab
        xz[i, :] = tmp.xz[2i-1, :]
        s[i] = tmp.s[2i-1]
        xz[i+n, :] = tmp.xz[2i, :]
        s[i+n] = tmp.s[2i]
    end
    return StabState(xz, s, n_stab)
end


function *(p1::PauliString, p2::PauliString)
    n = div(length(p1[2]), 2)
    s1, b1 = p1
    s2, b2 = p2
    s = s1 + s2
    for i in 1:n
        s += 2 * (b1[2i] && b2[2i-1])
    end
    s = m4(s)
    b = Bool[xor(b1[i], b2[i]) for i in 1:2n]
    return s, b
end


"""
xz[i, :] <- xz[j, :]
"""
function copy_row!(state, i, j)
    if i==j
        return nothing
    end
    xz, s, n_stab = flat(state)
    m, n = size(state)
    for k in 1:2n
        xz[i, k] = xz[j, k]
    end
    s[i] = s[j]
    return nothing
end


function erase_row!(state, i)
    xz, s, n_stab = flat(state)
    m, n = size(state)
    for k in 1:2n
        xz[i, k] = false
    end
    s[i] = 0
    return nothing
end


"""
xz[i] = xz[i] * xz[j]
"""
function row_sum!(state, i, j)
    xz, s, n_stab = flat(state)
    m, n = size(state)
    for k in 1:n
        s[i] += 2 * (xz[i, 2k] * xz[j, 2k-1])
        xz[i, 2k-1] ⊻= xz[j, 2k-1]
        xz[i, 2k] ⊻= xz[j, 2k]
    end
    s[i] += s[j]
    s[i] = m4(s[i])
    return nothing
end


@views function row_auto_fill!(state, i)
    xz, s, n_stab = flat(state)
    m, n = size(state)
    j = i>n ? i-n : i+n

    while !binary_symplectic_inner(xz[i, :], xz[j, :])
        rand!(xz[i, :])
    end
    for k in 1:m
        if k==i || k==j
            continue
        end
        if binary_symplectic_inner(xz[i, :], xz[k, :])
            row_sum!(state, i, k+n)
        end
        if binary_symplectic_inner(xz[i, :], xz[k+n, :])
            row_sum!(state, i, k)
        end
    end
    s[i] = Int(!is_herm(0, xz[i, :]))
    return nothing
end


"""
xz[i] = xz[i] * xz[j]
xz[j+n] = xz[i+n] * xz[j+n]
"""
function double_row_sum!(state, i, j)
    m, n = size(state)
    row_sum!(state, i, j)
    row_sum!(state, j+n, i+n)
    return nothing
end


# commute(x1::Bool, z1::Bool, x2::Bool, z2::Bool) = !xor(x1 * z2, x2 * z1)
# commute(x, y) = !binary_symplectic_inner(x, y)


function is_herm(s, xz)
    r = false
    for k in 1:2:length(xz)
        r ⊻= xz[k] * xz[k+1]
    end
    return !xor(isodd(s), r)
end


function random_clifford(n)
    xz = binary_random_symplectic_matrix(n)
    s = fill(0, 2n)
    for k in 1:2n
        h = is_herm(0, view(xz, k, :))
        s[k] = m4(2 * rand(Bool) + !h)
    end
    return Clifford(xz, s)
end

function random_cc_clifford(n)
    xz = binary_charge_conserving_symplectic_mat(n)
    s = fill(0, 2n)
    for k in 1:2n
        h = is_herm(0, view(xz, k, :))
        s[k] = m4(2 * rand(Bool) + !h)
    end
    s[1] = 0
    s[n+1] = 0
    return Clifford(xz, s)
end

function random_z2_clifford(n)
    # sample random n-site clifford gate that preserves \prod X
    ortho_b_mat = binary_random_orthogonal_matrix(2n)
    xz = binary_jordan_wigner_transform!(ortho_b_mat)
    xz[:, 2:2:end] .⊻= xz[:, 1:2:end]
    xz[1:2:end, :] .⊻= xz[2:2:end, :]
    s = fill(0, 2n)
    for i in 1:2n
        h = is_herm(0, view(xz, i, :))
        s[i] = (2 * rand(Bool) + h) % 4
    end
    tmp = 0, fill(false, 2n)
    for i in 1:2:2n
        tmp = tmp * (s[i], view(xz, i, :))
    end
    (tmp==2) && (s[1] = (s[1] + 2) % 4)
    return Clifford(xz, s)
end


"""
calculate the trace of a clifford unitary
"""
function abs_cliff_trace(cliff)
    xz, s = cliff.xz, cliff.s
    n = div(length(s), 2)
    mat = zeros(Bool, 2n, 4n)
    for i in 1:2n
        mat[i, i] = true
    end
    mat[1:2n, 2n+1:2n+2n] = xz

    xz_minus_i = copy(xz')
    for i in 1:2n
        xz_minus_i[i, i] ⊻= true
    end
    
    nullspace = binary_null_space(xz_minus_i)

    n_same = size(nullspace, 2)
    for i in 1:n_same
        tmp = 0, zeros(Bool, 4n)
        for j in 1:2n
            if nullspace[j, i]
                tmp = tmp * (s[j], mat[j, :])
            end
        end
#         @show tmp
        @assert tmp[1]==0 || tmp[1]==2
        @assert tmp[2][1:2n] == tmp[2][2n+1:4n]
        if tmp[1]==2
            return 0
        end
    end
    return 2^(n_same/2)
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


function clifford_action!(clifford, state, positions)
    n_act = length(positions)
    @assert size(clifford.xz, 1) == n_act * 2
    m, n = size(state)
    xz, s, n_stab = flat(state)
    indices = spin_to_binary_indices(positions)
    tmp_s = 0
    tmp_xz = fill(false, 2 * n_act)
    for k in 1:2n
        if (m < k <= n) || (n+m < k <= 2n)
            continue
        end
        tmp_s = 0
        tmp_xz .= false
        for j in 1:2n_act
            if xz[k, indices[j]]
                tmp_s += clifford.s[j]
                for k in 1:n_act
                    tmp_s += 2 * (tmp_xz[2k] * clifford.xz[j, 2k-1])
                    tmp_xz[2k-1] ⊻= clifford.xz[j, 2k-1]
                    tmp_xz[2k] ⊻= clifford.xz[j, 2k]
                end

            end
        end
        s[k] = m4(tmp_s + s[k])
        xz[k, indices] = tmp_xz
    end
    return nothing
end


"""
forced measurement on pure state
"""
@views function fps_measurement!(state, observable::PauliString, positions)
    m, n = size(state)
    xz, s, _ = flat(state)
    @assert m==n
    indices = spin_to_binary_indices(positions)
    tmp = 0
    for k in 1:n
        if binary_symplectic_inner(observable[2], xz[k, indices])
            if tmp == 0
                tmp = k
            else
                row_sum!(state, k, tmp) 
            end
        end
    end
    if tmp != 0
        for k in n+1:2n
            if binary_symplectic_inner(observable[2], xz[k, indices])
                row_sum!(state, k, tmp)
            end
        end
        copy_row!(state, tmp + n, tmp)
        erase_row!(state, tmp)
        s[tmp] = observable[1]
        xz[tmp, indices] .= observable[2]
    end
    return false
end


@views function measurement!(state, observable::PauliString, positions, record=true, forced=false)
    m, n = size(state)
    xz, s, _ = flat(state)
    indices = spin_to_binary_indices(positions)
    uc_stab_rows = Int[]
    uc_destab_rows = Int[]
    for k in 1:m
        if binary_symplectic_inner(observable[2], xz[k, indices])
            push!(uc_stab_rows, k)
        end
        if binary_symplectic_inner(observable[2], xz[k+n, indices])
            push!(uc_destab_rows, k)
        end
    end
    # case 1: random outcome, some stabilizer anti-commute with observable
    if !isempty(uc_stab_rows)
        i = uc_stab_rows[1]
        for j in uc_stab_rows[2:end]
            row_sum!(state, j, i)
        end
        for j in uc_destab_rows
            row_sum!(state, j+n, i)
        end
        # # case 1.1: recorded, n_stab unchanged
        if record
            copy_row!(state, i+n, i)
            xz[i, :] .= false
            s[i] = observable[1]
            xz[i, indices] = observable[2]
            r = forced ? false : rand(Bool)
            s[i] += 2 * r
            s[i] = m4(s[i])
            return r
        # case 1.2: unrecorded, n_stab - 1
        else
            copy_row!(state, i, m)
            copy_row!(state, i+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
            return false
        end
    else
        b = zeros(Bool, 2n)
        b[indices] .= observable[2]
        tmp = observable[1], b
        for i in uc_destab_rows
            tmp = tmp * (s[i], xz[i, :])
        end
        a = all(.!tmp[2])
        # case 2: deterministic outcome, n_stab unchanged
        if a
            if (!record) || (tmp[1] == 0)
                return false
            else
                return true
            end
        # case 3: random outcome, n_stab + 1 if recorded
        elseif !record
            return false
        else
            m += 1
            state.n_stab += 1
            s[m] = tmp[1]
            xz[m, :] = tmp[2]
            r = forced ? false : rand(Bool)
            s[m] += 2r
            s[m] = m4(s[m])
            row_auto_fill!(state, m+n)
            return r
        end
    end
end

entropy(state) = size(state, 2) - size(state, 1)

function entropy(state, rg)
    m, n = size(state)
    rg_bar = setdiff(1:n, rg)
    indices = spin_to_binary_indices(rg_bar)
    rk = binary_rank(state.xz[1:m, indices])
    return length(rg) - m + rk
end

function left_ee_on_all_cuts(state)
    m, n = size(state)
    mat = state.xz[1:m, 2n:-1:1]
    c_rks = binary_all_vertical_cut_ranks(mat)[2n:-2:2]
    ees = [i==n ? (n-m) : (i - m + c_rks[i+1]) for i in 0:n]
    return ees
end

function right_ee_on_all_cuts(state)
    m, n = size(state)
    mat = state.xz[1:m, :]
    c_rks = binary_all_vertical_cut_ranks(mat)[2:2:2n]
    ees = [i==0 ? (n-m) : (n - i - m + c_rks[i]) for i in 0:n]
    return ees
end

function mi_on_all_cuts(state)
    return left_ee_on_all_cuts(state) + right_ee_on_all_cuts(state) .- entropy(state)
end

function mutual_info(state, region1, region2)
    a = entropy(state, region1)
    b = entropy(state, region2)
    c = entropy(state, union(region1, region2))
    return a + b - c
end


function mutual_neg(state, region1, region2)
    m, n = size(state)
    l1, l2 = length(region1), length(region2)
    new_idx = cat(region1, region2, setdiff(1:n, union(region1, region2)), dims=1)
    new_idx = spin_to_binary_indices(new_idx)
    mat = state.xz[1:m, new_idx]
    epoints = binary_bidirectional_gaussian!(mat)
    mask = [i for i in 1:m if epoints[i, 2] <= 2(l1 + l2)]
    mat1 = @views mat[mask, 1:2l1]
    m = length(mask)
    K = zeros(Bool, m, m)
    for i in 1:m
        for j in 1:i
            @views K[i, j] = K[j, i] = binary_symplectic_inner(mat1[i, :], mat1[j, :])
        end
    end
    return div(binary_rank(K), 2)
end


function ap_negativity(state, a, b, l)
    @assert (a<b) && (a+l<=b)
    m, n = size(state)

    range_list = [1:2a, 2a+2l+1:2b, 2b+2l+1:2n]
    mat = fill(false, m, 2n)

    @views for i in 1:l
        mat[:, 2(2i-1)-1] = state.xz[1:m, 2(a+i)-1]
        mat[:, 2(2i-1)] = state.xz[1:m, 2(a+i)]

        mat[:, 2(2i)-1] = state.xz[1:m, 2(b+i)-1]
        mat[:, 2(2i)] = state.xz[1:m, 2(b+i)]
    end

    i = 4l
    @views for rg in range_list
        len = length(rg)
        if len>0
            mat[:, i+1:i+len] = state.xz[1:m, rg]
        end
        i += len
    end

    end_points = binary_bidirectional_gaussian!(mat)

    tmp = [(i, end_points[i, 2]) for i in 1:m if end_points[i, 2] <= 4l]
    sort!(tmp, by= x->x[2])
    new_order = Int[a[1] for a in tmp]
    end_points2 = end_points[new_order, :]
    m = length(new_order)

    mask_A = [0<i%4<3 for i in 1:4l]
    mat_A = mat[new_order, 1:4l][:, mask_A]

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
    ngs = [gk[2r]==0 ? 0 : div(rank_K[gk[2r]], 2) for r in 1:l]
    return ngs
end


@views function ap_mutual_info(state, a, b, l)
    m, n = size(state)
    mat = fill(false, m, 4l)

    for i in 1:l
        mat[:, 2(2i-1)-1] = state.xz[1:m, 2(a+i)-1]
        mat[:, 2(2i-1)] = state.xz[1:m, 2(a+i)]

        mat[:, 2(2i)-1] = state.xz[1:m, 2(b+i)-1]
        mat[:, 2(2i)] = state.xz[1:m, 2(b+i)]
    end

    mat_AB = mat
    mask = Bool[0<i%4<3 for i in 1:4l]
    mat_A = mat_AB[:, mask]
    mat_B = mat_AB[:, .!mask]

    rk_AB = binary_all_vertical_cut_ranks!(mat_AB)[2:2:end]
    rk_A = binary_all_vertical_cut_ranks!(mat_A)[2:2:end]
    rk_B = binary_all_vertical_cut_ranks!(mat_B)[2:2:end]

    mis = [rk_A[i] + rk_B[i] - rk_AB[2i] for i in 1:l]
    
    return mis
end


function strange_AB_mi(state, A)
    m, n = size(state)
    B = div(n - A, 2)
    mat = zeros(Bool, m, 2n)
    mat[1:m, 1:2A] = state.xz[1:m, 2B+1:2B+2A]
    for i in 1:B
        mat[1:m, 2A+4i-3] = state.xz[1:m, 2(B-i+1)-1]
        mat[1:m, 2A+4i-2] = state.xz[1:m, 2(B-i+1)]
        mat[1:m, 2A+4i-1] = state.xz[1:m, 2(A+B+i)-1]
        mat[1:m, 2A+4i-0] = state.xz[1:m, 2(A+B+i)]
    end
    ep = binary_bidirectional_gaussian!(mat)
    mis = zeros(Int, B)
    for i in 1:min(2A, m)
        l = div(ep[i, 1]+1, 2)
        r = div(ep[i, 2]+1, 2)
        if (l <= A) && (r > A)
            mis[div(r-A+1, 2):end] .+= 1
        end
    end
    return mis
end


function localizable_EE(state, A, B)
    m, n = size(state)
    E = setdiff(1:n, A, B)
    mat = state.xz[1:m, :]
    is_piv_row, pivs = binary_partial_gaussian!(mat, [2i for i in E])
    bA = spin_to_binary_indices(A)
    bB = spin_to_binary_indices(B)
    matA = mat[.!is_piv_row, bA]
    matB = mat[.!is_piv_row, bB]
    matAB = cat(matA, matB, dims=2)
    rkA = binary_rank(matA)
    rkB = binary_rank(matB)
    rkAB = binary_rank(matAB)
    return rkA + rkB - rkAB
end


function tri_mi(state, A, B, add_mi_AB)
    m, n = size(state)
    C = setdiff(1:n, A, B)
    a, b, c = length(A), length(B), length(C)
    bA = spin_to_binary_indices(A)
    bB = spin_to_binary_indices(B)
    bC = spin_to_binary_indices(C)
    mat = zeros(Bool, m, 2n)
    mat[1:m, 1:2a] = @view state.xz[1:m, bA]
    mat[1:m, 2a+1:2a+2c] = @view state.xz[1:m, bC]
    mat[1:m, 2a+2c+1:2a+2c+2b] = @view state.xz[1:m, bB]
    ep = binary_bidirectional_gaussian!(mat)
    mA, mB, mC, mAC, mCB = 0, 0, 0, 0, 0
    @inline inA(x) = 0 < x <= 2a
    @inline inC(x) = 2a < x <= 2a+2c
    @inline inB(x) = 2a+2c < x <= 2a+2b+2c
    for i in 1:m
        l, r = ep[i,:]
        if inA(r)
            mA += 1
            mAC += 1
        elseif inC(l) && inC(r)
            mC += 1
            mAC += 1
            mCB += 1
        elseif inB(l)
            mB += 1
            mCB += 1
        elseif inA(l) && inC(r)
            mAC+= 1
        elseif inC(l) && inB(r)
            mCB+= 1
        end
    end
    tri = mCB + mAC - mC - m
    if add_mi_AB
        tri += (a+b-entropy(state, union(A, B))) - mA - mB
    end
    return tri
end