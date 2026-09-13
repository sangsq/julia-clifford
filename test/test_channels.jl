function scramble!(channel)
    m, n = size(channel)
    cliff = random_clifford(n)
    clifford_action_on_channel!(cliff, channel, 1:n)
    return nothing
end


function test_channel_decompose()
    p = 0.25
    q = 1
    n = 64
    m = Int(p*n)

    ch = identity_channel(m)
    @assert channel_decompose(ch) == (0, 0, m)
    add_qubits!(ch, Int((1-p)*n))
    test_state(ch.choi)
    @assert channel_decompose(ch) == (0, 0, m)
    scramble!(ch)
    @assert channel_decompose(ch) == (0, 0, m)
    for i in 1:Int(q*n)
        depolarize!(ch, i)
    end
    test_state(ch.choi)
    @assert channel_decompose(ch) == (m, 0, 0)
end


function test_z_dephase()
    n = 8
    ch = identity_channel(n)
    scramble!(ch)
    for i in 1:n
        z_dephase!(ch, i)
    end
    test_state(ch.choi)
    @assert channel_decompose(ch) == (0, n, 0)
end
