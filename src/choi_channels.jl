"""
Here a (m qubit -> n qubit) stabilizer channel is represented by its vectorized Choi state, which is a 2(n+m) qubit pure stabilizer state.
The ordering of legs is: | n out-left | n out-right | m in-left | m in-right |
"""
mutable struct StabChannel
    choi::StabState
    m::Int # number of in qubits
    n::Int # number of out qubits
end

size(ch::StabChannel) = ch.m, ch.n


function identity_channel(n)
    choi = epr_pairs(2n)
    return StabChannel(choi, n, n)
end


function clifford_action_on_channel!(clifford, channel, positions)
    choi = channel.choi
    s = choi.s
    m, n = size(channel)

    clifford_action!(clifford, choi, positions)
    for i in 1:2(m+n)
        if isodd(s[i])
            s[i] = m4(s[i] + 2)
        end
    end
    clifford_action!(clifford, choi, positions .+ n)
    for i in 1:2(m+n)
        if isodd(s[i])
            s[i] = m4(s[i] + 2)
        end
    end

    return nothing
end


function depolarize!(channel::StabChannel, i)
    choi = channel.choi
    m, n = size(channel)
    fps_measurement!(choi, (0, Bool[1, 0, 1, 0]), [i, i+n])
    fps_measurement!(choi, (0, Bool[0, 1, 0, 1]), [i, i+n])
    return nothing
end


function z_dephase!(channel, i)
    m, n = size(channel)
    choi = channel.choi
    fps_measurement!(choi, (0, Bool[1, 0, 1, 0]), [i, i+n])
    return nothing
end


function z_damp!(channel, i)
    m, n = size(channel)
    choi = channel.choi
    depolarize!(channel, i)
    fps_measurement!(choi, (0, Bool[1, 0]), [i])
    fps_measurement!(choi, (0, Bool[1, 0]), [i+n])
    return nothing
end

@views function add_qubits!(channel, l)
    m, n = size(channel)
    choi = channel.choi
    N = 2n+2m
    M = N+2l
    new_xz = zeros(Bool, 2M, 2M)
    for (r_old, r_new) in ((1:N, 1:N), (N+1:2N, M+1:M+N))
        new_xz[r_new, 1:2n] = choi.xz[r_old, 1:2n]
        new_xz[r_new, 2n+2l+1:4n+2l] = choi.xz[r_old, 2n+1:4n]
        new_xz[r_new, 4n+4l+1:4n+4l+4m] = choi.xz[r_old, 4n+1:4n+4m]
    end
    for i in 1:l
        new_xz[N+i, 2n+2i] = true
        new_xz[N+l+i, 4n+2l+2i] = true
        new_xz[M+N+i, 2n+2i-1] = true
        new_xz[M+N+l+i, 4n+2l+2i-1] = true
    end
    new_s = zeros(Int, 2M)
    new_s[1:N] .= choi.s[1:N]
    new_s[M+1:M+N] .= choi.s[N+1:2N]
    choi.xz = new_xz
    choi.s = new_s
    choi.n_stab = M
    channel.n += l
    return nothing
end

    


function channel_decompose(channel)
    m, n = size(channel)
    choi = channel.choi
    rg1, rg2, rg3, rg4 = 1:n, n+1:2n, 2n+1:2n+m, 2n+m+1:2n+2m
    n_discard = mutual_neg(choi, rg3, rg4)
    n_dph = mutual_info(choi, rg3, rg4) - 2n_discard
    n_id = mutual_neg(choi, rg1, rg3)
    @assert mutual_neg(choi, rg2, rg4) == n_id
    return n_discard, n_dph, n_id
end