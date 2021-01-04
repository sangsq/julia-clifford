include("cliff5.jl")

function random_Z2_clifford(n)
    ortho_b_mat = binary_random_orthogonal_matrix(2n)
    sp_b_mat = binary_jordan_wigner_transform!(ortho_b_mat)
    sp_b_mat[:, 2:2:end] .⊻= sp_b_mat[:, 1:2:end] 
    p_mat = to_stablizer_state(sp_b_mat)
    r = ()
    for i in 1:n
        y = (p_mat[2i-1, :]...,)
        z = (p_mat[2i, :]...,)
        r = (y .* z, y, z, r...)
    end
    return r
end


function random_Z2_2clifford()
    even_2paulis = [(II, X), (X, II), (Y, Y), (Y, Z), (Z, Y), (Z, Z)]
    odd_2paulis = [(II, Y), (II, Z), (X, Y), (X, Z), (Y, II), (Y, X), (Z, II), (Z, X)]

    a = even_2paulis
    b = odd_2paulis

    x1 = rand(a)
    z1 = nothing

    while true
        z1 = rand(b)
        if !commute(z1, x1)
            break
        end
    end

    a = filter(e -> commute(e, x1) && commute(e, z1), a)    
    b = filter(e -> commute(e, x1) && commute(e, z1), b)

    x2 = rand(a)
    z2 = nothing

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

function measurement(state, observable::PauliString, positions)
    m, n = size(state)
    uncommute_rows = [i for i in 1:m if !commute(observable, state[i, positions])]
    if isempty(uncommute_rows)
        return state
    end
    for row in uncommute_rows[2:end]
        state[row, :] .*= state[uncommute_rows[1], :]
    end
    row = uncommute_rows[1]
    state[row, :] .= II 
    state[row, positions] = observable
    return state
end

function X_measurement!(state, posi)
    M, N = size(state)
    uncommute_rows = [i for i in 1:M if !commute(X, state[i, posi])]
    if isempty(uncommute_rows)
        return state
    end
    for row in uncommute_rows[2:end]
        state[row, :] .*= @view state[uncommute_rows[1], :]
    end
    row = uncommute_rows[1]
    state[row, :] .= II 
    state[row, posi] = X
    return state
end


function ZZ_measurement!(state, posi1, posi2)
    M, N = size(state)
    uncommute_rows = [i for i in 1:M if !commute((Z, Z), (state[i, posi1], state[i, posi2]))]
    if isempty(uncommute_rows)
        return state
    end
    for row in uncommute_rows[2:end]
        state[row, :] .*= @view state[uncommute_rows[1], :]
    end
    row = uncommute_rows[1]
    state[row, :] .= II
    state[row, [posi1, posi2]] = Paulis[Z, Z]
    return state
end


function cal_phi(state, connected)
    M, N = size(state)
    mat1 = (state .== X)
    mat2 = (state .== Y)
    mat = xor.(mat1, mat2)

    dic = Dict{Array{Bool, 1}, Int}()
    for s in 1:N
        tmp = mat[:, s]
        if tmp in keys(dic)
            dic[tmp] += 1
        else
            dic[tmp] = 1
        end
    end

    phi = sum([count * (count - 1) / 2 for count in values(dic)])

    if connected
        tmp = [false for _ in 1:M]
        if tmp in keys(dic)
            phi -= dic[tmp] * (dic[tmp] - 1) / 2
        end
    end

    return phi / N
end   


function mixed_gate(r, p, sym)
    function tmp(state, posi1, posi2)
        if rand() > p
            cliff = sym ? random_Z2_2clifford() : random_2clifford()
            cliff2_action(cliff, state, posi1, posi2)
        else
            if rand() > r
                X_measurement!(state, posi1)
            else
                ZZ_measurement!(state, posi1, posi2)
            end
        end
        return state
    end
    return tmp
end


function mixed_gate_on4(r, p, sym)
    function tmp(state, posi1, posi2, posi3, posi4)
        if rand() > p
            cliff = sym ? random_Z2_clifford(4) : random_clifford(4)
            clifford_action_on_state(cliff, state, [posi1, posi2, posi3, posi4])
        else
            if rand() > r
                X_measurement!(state, posi1)
            else
                ZZ_measurement!(state, posi1, posi2)
            end
            if rand() > r
                X_measurement!(state, posi3)
            else
                ZZ_measurement!(state, posi3, posi4)
            end
        end
        return state
    end
    return tmp
end


function free_mixed_gate(r, p, sym)
    swap12 = ((X, II), (Z, II), (Y, II), (II, X), (II, Y), (II, Z))
    swap23 = ((Z, Y), (Y, II), (X, Y), (Y, Z), (II, Y), (Y, X))
    function tmp(state, posi1, posi2)
        posi2 == 1 && return state # enforce the open boundary condition
        if rand() > p
            cliff = rand() > 0.5 ? swap12 : swap23
            cliff2_action(cliff, state, posi1, posi2)
        else
            if rand() > r
                X_measurement!(state, posi1)
            else
                ZZ_measurement!(state, posi1, posi2)
            end
        end
        return state
    end
    return tmp
end


function multi_point_sym_gate(p)
    function tmp(state, positions)
        if rand() > p
            cliff = random_Z2_clifford(length(positions))
            clifford_action_on_state(cliff, state, positions)
        else
            [ZZ_measurement!(state, positions[p], positions[p+1]) for p in 1:(length(positions)-1)]
        end
        return state
    end
    return tmp
end
            

function adam_gate(p, sym, dilution=1)
    function tmp(state, posi1, posi2)
        if rand() < dilution
            cliff = sym ? random_Z2_2clifford() : random_2clifford()
            cliff2_action(cliff, state, posi1, posi2)
        end

        if rand() < p * dilution
            X_measurement!(state, posi1)
        end
        
        if rand() < p * dilution
            X_measurement!(state, posi2)
        end
        return state
    end
    return tmp
end



function cor_to_idx(cor, shape)
    i, j = cor
    i -= 1
    j -= 1
    n1, n2 = shape
    idx = i * n2 + j
    return idx + 1
end


function idx_to_cor(idx, shape)
    n1, n2 = shape
    idx -= 1
    i = div(idx, n2)
    j = idx % n2
    return i + 1, j + 1
end


function apply_gate_2d!(state, gate, shape, cors)
    positions = [cor_to_idx(c, shape) for c in cors]
    gate(state, positions)
end

function zz_adam_gate(p, sym, dilution=1)
    function tmp(state, posi1, posi2)
        m, n = size(state)
        if rand() < dilution
            cliff = sym ? random_Z2_2clifford() : random_2clifford()
            cliff2_action(cliff, state, posi1, posi2)
        end

        if rand() < p * dilution
            if posi1 == 1
                ZZ_measurement!(state, n, posi1)
            else
                ZZ_measurement!(state, posi1-1, posi1)
            end
        end
        
        if rand() < p * dilution
            ZZ_measurement!(state, posi1, posi2)
        end
        return state
    end
    return tmp
end


function square_region(x_range, y_range, shape)
    tmp = []
    for idx1 in x_range
        for idx2 in y_range
            cor = cor_to_idx((idx1, idx2), shape)
            push!(tmp, cor)
        end
    end
    return tmp
end