using Random
using Profile
using LinearAlgebra
import Base:show, *, length, iterate

include("binary_linalg.jl")

# Random.seed!(1111)

@enum Paulis::UInt8 II=0 X=1 Y=2 Z=3

length(::Paulis) = 1
iterate(p::Paulis) = (p, nothing)
iterate(p::Paulis, ::Nothing) = nothing

function paulis_empty(dims...)
    ele = 1
    for i in dims
        ele *= i
    end
    return reshape(Paulis[II for _ in 1:ele], dims)
end

function bell_pairs(L)
    state = paulis_empty(2L,2L)
    for i in 1:L
        state[2*i - 1, i] = state[2*i - 1, i + L] = X
        state[2*i, i] = state[2*i, i + L] = Z
    end
    return state
end

const PauliString = Array{Paulis, 1}
PauliString(d1::Paulis, d2::Paulis) = Paulis[d1, d2]
PauliString(t::Tuple) = Paulis[t...]

const StablizerState = Array{Paulis, 2}

function commute(x::Paulis, y::Paulis)
    x==II && return true
    y==II && return true
    x==y && return true
    return false
end

commute(x::PauliString, y::PauliString) = !xor(.!commute.(x, y)...)
commute(x::NTuple{N, Paulis}, y::NTuple{N, Paulis}) where N = !xor(.!commute.(x, y)...)

*(x::Paulis, y::Paulis) = Paulis(xor(UInt8(x), UInt8(y)))

const CZ = ((X, Z), (Y, Z), (Z, II), (Z, X), (Z, Y), (II, Z))
const IID = ((X, II), (Y, II), (Z, II), (II, X), (II, Y), (II, Z))


function none_zero_paulis(n, _called=false)
    if n==1
        return [(II,), (X, ), (Y, ), (Z, )]
    else
        tmp = none_zero_paulis(n-1, true)
        tmpp = []
        for k in [II, X, Y, Z]
            a = [(k, l...) for l in tmp]
            push!(tmpp, a)
        end
    end
    if _called
        return [tmpp[1]..., tmpp[2]..., tmpp[3]..., tmpp[4]...]
    else
        return [tmpp[1][2:end]..., tmpp[2]..., tmpp[3]..., tmpp[4]...]
    end
end


function random_clifford(n)
    a = none_zero_paulis(n)
    default = a[1]
    result = NTuple{n, Paulis}[default for _ in 1:3n]
    for k in 1:n
        x = rand(a)
        z = rand(a)
        while commute(x, z)
            z = rand(a)
        end
        result[3k-2], result[3k-1], result[3k]  = x, x .* z, z
        a = filter(e -> commute(e, x) && commute(e, z), a)
    end
    return result
end


function random_2clifford()
    none_zero_2paulis = [
                (II, X), (II, Y), (II, Z),
        (X, II), (X, X), (X, Y), (X, Z),
        (Y, II), (Y, X), (Y, Y), (Y, Z),
        (Z, II), (Z, X), (Z, Y), (Z, Z)
        ]

    a = none_zero_2paulis
    x1 = rand(a)
    z1 = (II, II)
    while true
        z1 = rand(a)
        if !commute(z1, x1)
            break
        end
    end

    b = filter(e -> commute(e, x1) && commute(e, z1), a)
    x2 = rand(b)
    z2 = (II, II)
    while true
        z2 = rand(b)
        if !commute(z2, x2)
            break
        end
    end

    y1 = x1 .* z1
    y2 = x2 .* z2

    return (x1, y1, z1, x2, y2, z2)
end


function random_1clifford()
    a = (X, Y, Z)
    x = rand(a)
    z = x
    while true
        z = rand(a)
        !commute(x, z) && break
    end
    return (x, x * z, z)
end


function clifford_action_on_pauli_string(clifford_gate, pauli_string)
    result = [II for _ in pauli_string]
    for k in 1:length(pauli_string)
        p = UInt8(pauli_string[k])
        p==0 && continue
        img = clifford_gate[p + 3 * (k-1)]
        result .*= img
    end
    return result
end


function clifford_action_on_state(clifford_gate, state, positions)
    for i in 1 : size(state)[1]
        stablizer = view(state, i, :)
        new_paulis = clifford_action_on_pauli_string(clifford_gate, stablizer[positions])
        stablizer[positions] = new_paulis
    end
    return state
end


function cliff2_action(cliff, p1, p2)
    N, M = UInt(p1), UInt(p2)
    r1 = (N == 0) ? (II, II) : cliff[N]
    r2 = (M == 0) ? (II, II) : cliff[3 + M]
    return r1 .* r2
end
            
function cliff2_action(cliff, state::StablizerState, posi1, posi2)
    for i in 1 : size(state)[1]
        stablizer = view(state, i, :)
        pauli2 = cliff2_action(cliff, stablizer[posi1], stablizer[posi2])
        stablizer[posi1], stablizer[posi2] = pauli2
    end
    return state
end

function almost_id(n, i, p)
    tmp = Paulis[II for i in 1:n]
    tmp[i] = p
    return tmp
end

all_up(n) = cat([almost_id(n, i, Z) for i in 1:n]..., dims=2)
all_plus(n) = cat([almost_id(n, i, X) for i in 1:n]..., dims=2)

function to_bit(p::Paulis)
    p == II && return false, false
    p == X && return true, false
    p == Y && return true, true
    p == Z && return false, true
end

function to_pauli(b1, b2)
    b1 == false && b2 == false && return II 
    b1 == true && b2 == false && return X 
    b1 == true && b2 == true && return Y 
    b1 == false && b2 == true && return Z 
end

function to_binary_matrix(state)
    M, N = size(state)

    result = zeros(Bool, (M, N * 2))
#     result = BitArray(undef, (M, N * 2))

    for i in 1:M
        for j in 1:N
            result[i, 2j-1], result[i, 2j] = to_bit(state[i, j])
        end
    end
    return result
end

function to_stablizer_state(bmat)
    M, N = size(bmat)
    N = N ÷ 2
    state = paulis_empty(M, N)
    for i in 1:M
        for j in 1:N
            state[i,j] = to_pauli(bmat[i, 2j - 1], bmat[i, 2j])
        end
    end
    return state
end

function sub_area_state(state::StablizerState, sub_area)
    M, N = size(state)
    L = size(sub_area)[1]

    binary_sub_area = zeros(Int, 2L)
    binary_sub_area[1:2:2*L] .+= 2 .* sub_area .- 1
    binary_sub_area[2:2:2*L] .+= 2 .* sub_area
    
    m = transpose(to_binary_matrix(state))
    m = cat(m, zeros(Bool, (2N, 2L)), dims=2)
    m[binary_sub_area, M + 1 : M + 2L] += I
    null_space = binary_null_space(m)[M + 1: M + 2L, :]
    new_state = to_stablizer_state(transpose(null_space))
    return new_state
end


function sign_mat(state)
    n_row = size(state)[1]
    mat = zeros(Bool, (n_row, n_row))
    for i in 1:n_row
        for j in 1:(i-1)
            if !commute(state[i, :], state[j, :])
                mat[i, j] = mat[j, i] = true
            end
        end
    end
    return mat
end

function negativity(state, sub_area)
    return binary_rank(sign_mat(state[:, sub_area]))
end

rk(state) = binary_rank(to_binary_matrix(state))

function entropy(state)
    M, N = size(state)
    return N - M
end

function bipartite_entropy(state, sub_area)
    M, N = size(state)
    sub_size = length(sub_area)
    sub_area_c = setdiff(1:N, sub_area)

    c_rk = rk(state[:, sub_area_c])
    return sub_size - M + c_rk
end


function pure_state_bipartite_entropy(state, sub_area)
    # Make use of the fact that for pure state S_A = S_B
    M, N = size(state)
    sub_size = length(sub_area)
    sub_area_c = setdiff(1:N, sub_area)

    if sub_size < div(N, 2)
        sub_area, sub_area_c = sub_area_c, sub_area
        sub_size = N - sub_size
    end

    c_rk = rk(state[:, sub_area_c])
    ee = sub_size - M + c_rk

    return ee
end

function pure_ee_across_all_cuts(state, cut_point)
    m, n = size(state)
    mat = to_binary_matrix(state[:, 1:cut_point])
    ees = Array(-1:-1:-cut_point)
    pivs, _ = binary_uppertrianglize!(mat)

    rho_left = zeros(Int, n)
    for b_piv in pivs
        piv = div(b_piv + 1, 2)
        rho_left[piv] += 1
    end

    current_sum = 0
    for i in 1:cut_point
        current_sum += rho_left[i]
        ees[i] += current_sum
    end

    return ees
end


function two_point_correlation_square(state, i, j, connected=true, d_i=Z, d_j=Z)
    M, N = size(state)
    a = [commute(p, d_i) for p in state[:, i]]
    b = [commute(p, d_j) for p in state[:, j]]
    tmp = [true for _ in 1:M]
    result = Int(a == b) - Int(a == b == tmp) * Int(connected)
    return result
end


function two_point_mutual_info(state, i, j)
    M_AB = view(state, :, [i, j])
    M_A = view(state, :, [i])
    M_B = view(state, :, [j])
    r_A, r_B, r_AB = rk(M_A), rk(M_B), rk(M_AB)
    return r_A + r_B - r_AB
end


function mutual_info(state, regionA, regionB)
    M_AB = view(state, :, union(regionA, regionB))
    M_A = view(state, :, regionA)
    M_B = view(state, :, regionB)
    r_A, r_B, r_AB = rk(M_A), rk(M_B), rk(M_AB)
    return r_A + r_B - r_AB
end


function mutual_neg(state, A, B)
    sub_state = sub_area_state(state, union(A, B))
    mn = negativity(sub_state, 1:length(A))
    return mn
end


function several_mutual_info(state, a, rd_list_a, b, rd_list_b)
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
    end

    return result
end


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


# function antipodal_negativity(state, rd_list)
#     state = copy(state)
#     m, n = size(state)
#     @assert m==n
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
    sort!(tmp2, by= x->x[2])
    new_order = Int[a[1] for a in tmp2]
    mat, end_points = mat[new_order, 1:4l], end_points[new_order, :]
    m = length(new_order)

    mask_A = [0<i%4<3 for i in 1:4l]
    mat_A = mat[:, mask_A]

    gk = fill(m, 2l)
    j = 1
    for i in 1:m
        k = end_points[i, 2]
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


function ap_mutual_info(state, a, b, l)
    m, n = size(state)
    tmp = fill(II, m, 2l)
    @views tmp[:, 1:2:2l] = state[:, a+1:a+l]
    @views tmp[:, 2:2:2l] = state[:, b+1:b+l]
    
    mat_AB = to_binary_matrix(tmp)
    mask = Bool[0<i%4<3 for i in 1:4l]
    mat_A = mat_AB[:, mask]
    mat_B = mat_AB[:, .!mask]

    rk_AB = binary_all_vertical_cut_ranks!(mat_AB)[2:2:end]
    rk_A = binary_all_vertical_cut_ranks!(mat_A)[2:2:end]
    rk_B = binary_all_vertical_cut_ranks!(mat_B)[2:2:end]

    mis = [rk_A[i] + rk_B[i] - rk_AB[2i] for i in 1:l]
    
    return mis
end