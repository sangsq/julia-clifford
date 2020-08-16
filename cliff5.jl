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
    binary_sub_area[1:2:2*L] += 2 .* sub_area .- 1
    binary_sub_area[2:2:2*L] += 2 .* sub_area
    
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
    if ee < 0
        display(state)
    end
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


function antipodal_negativity(state, rd_list)
    state = copy(state)
    n = size(state, 1)
    mid = div(n, 2)
    state[:, 1:2:end], state[:, 2:2:end] = state[:, 1:mid], state[:, (mid+1):end]
    
    mat = to_binary_matrix(state)
    end_points = binary_bidirectional_gaussian!(mat)
    tmp = [(i, end_points[i, 2]) for i in 1:n]
    sort!(tmp, by= x-> x[2])
    new_order = [a[1] for a in tmp]
    mat, end_points = mat[new_order, :], end_points[new_order, :]

    mask_A = [0<i%4<3 for i in 1:2n]
    mat_A = mat[:, mask_A]

    gk = [n for _ in 1:n]
    j = 1
    for i in 1:n
        k = end_points[i, 2]
        spin_k = div(k+1, 2)
        gk[j:spin_k-1] .= i-1
        j = spin_k
    end

    K = zeros(Bool, n, n)
    for i in 1:n
        for j in 1:i
            K[i, j] = K[j, i] = binary_symplectic_inner(mat_A[i, :], mat_A[j, :])
        end
    end

    ### code from yaodong ###
    rank_K = Int[]
	pivot    = [0 for i in 1:n]
    is_pivot = [false for i in 1:n]
    
	r_tmp = 0
	for i = 1:n
		for j = 1:i-1
			if (is_pivot[j] == 0) && (K[j,i] == 1)
				if pivot[i] == 0
					pivot[i] = j
					is_pivot[j] = 1
					r_tmp += 1
				else
					for k = 1:n
						K[j, k] = xor(K[j,k], K[pivot[i], k])
					end
				end
			end
		end

		for j = 1:i
			if K[i, j] == 1
				if pivot[j] != 0
					for k = 1:n
						K[i, k] = xor(K[i,k], K[pivot[j], k])
					end
				else
					pivot[j] = i
					is_pivot[i] = 1
					r_tmp += 1

					break
				end
			end
		end

		push!(rank_K, r_tmp)
	end
    ### code from yaodong ###

    # ngs = [binary_rank(K[1:gk[2r], 1:gk[2r]]) for r in rd_list]
    ngs = [gk[2r]==0 ? 0 : rank_K[gk[2r]] for r in rd_list]
    return ngs
end

# n = 200
# s = 100
# a = all_plus(n)
# for _ in 1:s
#     for i in 1:2:n-1
#         cliff2_action(random_2clifford(), a, i, i+1)
#     end
#     for i in 2:2:n-1
#         cliff2_action(random_2clifford(), a, i, i+1)
#     end
# end
# r_list = 2:2:100

# mid = div(n,2)
# # u = antipodal_negativity(a, r_list)
# # tu = [negativity(sub_area_state(a, union(1:r, mid .+ (1:r))),1:r) for r in r_list]
# # @show u - tu
# v = antipodal_mutual_info(a, r_list)
# tv = [mutual_info(a, 1:r, mid.+(1:r)) for r in r_list]
# @show (v - tv)




# function negativity_acc(s::state, sizeA, sizeB, w, rp)
# 	n  = s.n
# 	nG = s.m

# 	K    = Matrix{Bool}(nG, nG)
# 	for i = 1:nG, j = 1:nG
# 		K[i,j] = 0
# 	end

# 	for pos in w[1:sizeA]
# 		for i = 1:nG, j = i+1:nG
# 			if mod(s.x[i+n, pos]*s.z[j+n, pos]+s.z[i+n, pos]*s.x[j+n, pos], 2) == 1
# 				K[i,j] = mod(K[i,j]+1, 2)
# 				K[j,i] = mod(K[j,i]+1, 2)
# 			end
# 		end
#     end
    
# 	rank_K = []
# 	pivot    = [0 for i in 1:nG]
# 	is_pivot = [0 for i in 1:nG]

# 	r_tmp = 0

# 	for i = 1:nG
# 		for j = 1:i-1
# 			if (is_pivot[j] == 0) && (K[j,i] == 1)
# 				if pivot[i] == 0
# 					pivot[i] = j
# 					is_pivot[j] = 1
# 					r_tmp += 1
# 				else
# 					for k = 1:nG
# 						K[j, k] = mod(K[j,k] + K[pivot[i], k], 2)
# 					end
# 				end
# 			end
# 		end

# 		for j = 1:i
# 			if K[i, j] == 1
# 				if pivot[j] != 0
# 					for k = 1:nG
# 						K[i, k] = mod(K[i,k] + K[pivot[j], k], 2)
# 					end
# 				else
# 					pivot[j] = i
# 					is_pivot[i] = 1
# 					r_tmp += 1

# 					break
# 				end
# 			end
# 		end

# 		#@show i, r_tmp
# 		push!(rank_K, r_tmp)
# 	end

# 	#@show rank_K

# 	res = []
# 	#res_tmp = []
# 	now = 1
# 	for b = sizeA+1:sizeA+sizeB
# 		#@show b
# 		last_now = now
# 		while now <= nG && rp[now] <= b
# 			now += 1
# 		end

# 		#@show now

# 		if now == 1
# 			push!(res, 0)
# 		else
# 			push!(res, rank_K[now-1])
# 		end
# 	end
# 	return res
# end