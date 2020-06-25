using Random
using Profile
using LinearAlgebra
import Base:show, *, length, iterate

include("binary_linalg.jl")

Random.seed!(1111)

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

