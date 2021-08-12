include("channels2.jl")


function scramble!(channel)
    m, n = size(channel)
    cliff = random_clifford(n)
    clifford_action_on_channel!(cliff, channel, 1:n)
    return nothing
end


p = 0.25
q = 1
n = 64

ch = identity_channel(Int(p*n))
@show channel_decompose(ch)
add_qubits!(ch, Int((1-p)*n))
@show channel_decompose(ch)
scramble!(ch)
@show channel_decompose(ch)
for i in 1:Int(q*n)
    depolarize!(ch, i)
end
@show channel_decompose(ch)
