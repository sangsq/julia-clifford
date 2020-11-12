include("cliff6.jl")

n = 3
k = 2
state = all_up(n)
cliff = random_clifford(k)

clifford_action!(cliff, state, (1,2))
