using Random
using LinearAlgebra
import Base:show, *, length, iterate, size

include("binary_linalg.jl")

mutable struct StabState
    xz::Array{Bool, 2} # (2n, 2n) shape, on each site 10->X, 01->Z, 11->XZ, 1 to n row stab, (n+1) to 2n row destab
    s::Array{Int, 1} # shape (2n,), {0, 1, 2, 3} -> {1, i, -1, -i}
    n_stab::Int # number of stablizer for the state
end

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

const PauliString = Tuple{Int, <:AbstractArray{Bool}}

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


function random_state(n, n_stab)
    tmp = random_clifford(n)
    xz = zeros(Bool, 2n, 2n)
    s = zeros(Int, 2n)
    @views for i in 1:n_stab
        xz[i, :] = tmp.xz[2i-1, :]
        s[i] = tmp.s[2i-1, :]
        xz[i+n, :] = tmp.xz[2i, :]
        s[i+n] = tmp.s[2i, :]
    end
    return StabState(xz, s, n_stab)
end





function *(p1::PauliString, p2::PauliString)
    n = div(length(p1), 2)
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


function single_row_sum!(state, i, j)
    xz, s, n_stab = flat(state)
    m, n = size(state)
    for k in 1:n
        s[i] += 2 * (xz[i, 2k] * xz[j, 2k-1])
        xz[i, 2k-1] ⊻= xz[j, 2k-1]
        xz[i, 2k] ⊻= xz[j, 2k]
    end
    s[i] = m4(s[i])
    return nothing
end


"""
xz[i] = xz[i] * xz[j]
dxz[j] = dxz[i] * dxz[j]
"""
function row_sum!(state, i, j)
    m, n = size(state)
    single_row_sum!(state, i, j)
    single_row_sum!(state, j+n, i+n)
    return nothing
end


# commute(x1::Bool, z1::Bool, x2::Bool, z2::Bool) = !xor(x1 * z2, x2 * z1)
# commute(x, y) = !binary_symplectic_inner(x, y)


function is_herm(s, xz)
    r = false
    for i in 1:2:length(xz)
        r ⊻= xz[i] * xz[i+1]
    end
    return !xor(isodd(s), r)
end


function random_clifford(n)
    xz = binary_random_symplectic_matrix(n)
    s = fill(0, 2n)
    for i in 1:2n
        h = is_herm(0, view(xz, i, :))
        s[i] = m4(2 * rand(Bool) + !h)
    end
    return Clifford(xz, s)
end


function spin_to_binary_indices(spin_indices)
    n = length(spin_indices)
    indices = fill(0, 2 * n)
    for i in 1:n
        indices[2i-1] = 2 * spin_indices[i] - 1
        indices[2i] = 2 * spin_indices[i]
    end
    return indices
end


function clifford_action!(clifford, state, positions)
    n_act = length(positions)
    @assert size(clifford.xz, 1) == n_act * 2
    m, n = size(state)
    xz, s, n_stab = flat(state)
    indices = spin_to_binary_indices(positions)
    for i in union(1:m, 1+n:m+n)
        tmp_s = 0
        tmp_xz = fill(false, 2 * n_act)
        for j in 1:2n_act
            if xz[i, indices[j]]
                tmp_s += clifford.s[j]
                for k in 1:n_act
                    tmp_s += 2 * (tmp_xz[2k] * clifford.xz[j, 2k-1])
                    tmp_xz[2k-1] ⊻= clifford.xz[j, 2k-1]
                    tmp_xz[2k] ⊻= clifford.xz[j, 2k]
                end
            end
        end
        s[i] = m4(tmp_s + s[i])
        xz[i, indices] = tmp_xz
    end
    return nothing
end


function measurement!(state, observable::PauliString, positions, record=true)
    m, n = size(state)
    xz, s, _ = flat(state)
    indices = spin_to_binary_indices(positions)
    uc_stab_rows = Int[]
    uc_destab_rows = Int[]
    for i in 1:m
        if binary_symplectic_inner(observable[2], @view xz[i, indices])
            push!(uc_stab_rows, i)
        end
        if binary_symplectic_inner(observable[2], @view xz[i+n, indices])
            push!(uc_destab_rows, i)
        end
    end
    # case 1: random outcome, some stabilizer anti-commute with observable
    if !isempty(uc_stab_rows)
        i = uc_stab_rows[1]
        for j in uc_stab_rows[2:end]
            row_sum!(state, i, j)
        end
        # case 1.1: recorded, n_stab unchanged
        if record
            s[i+n] = s[i]
            xz[i+n, :] .= @view xz[i, :]
            xz[i, :] .= false
            s[i], xz[i, indices] = observable
            r = rand(Bool)
            s[i] += 2 * r
            s[i] = m4(s[i])
            return r
        # case 1.2: unrecorded, n_stab - 1
        else
            s[i] = s[m]
            xz[i, :] = @view xz[m, :]
            s[i+n] = s[m+n]
            xz[i+n, :] = @view xz[m+n, :]
            s[m] = 0
            xz[m, :] .= false
            s[m+n] = 0
            xz[m+n, :] .= false
            state.n_stab -= 1
            return false
        end
    else
        tmp = observable
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
        # case 3: random outcome, n_stab + 1
        elseif !record
            return false
        else
            state.n_stab += 1
            m += 1
            s[m], xz[m, indices] = tmp
            r = rand(Bool)
            s[m] += 2r
            s[m] = m4(s[m])
            while !binary_symplectic_inner(xz[m+n, :], xz[m, :])
                xz[m+n, :] = rand(Bool, 2n)
            end
            s[m+n] = Int(!is_herm(0, @view xz[m+n, :]))
            for i in 1:m-1
                if binary_symplectic_inner(xz[m+n, :], xz[i, :])
                    single_row_sum!(state, m+n, i+n)
                end
                if binary_symplectic_inner(xz[m+n, :], xz[i+n, :])
                    single_row_sum!(state, m+n, i)
                end
            end
            return r
        end
    end
end
