using BenchmarkTools
include("cliff5.jl")
include("sym_cliff.jl")

# Random.seed!(1)
# let
#     for _ in 1:10000
#         n = 10
#         m = 7
#         a = 2
#         b = 6
#         l = 4

#         mat = binary_random_symplectic_matrix(n)
#         state = to_stablizer_state(mat[2:2:2m, :])
#         result = ap_mutual_info(state, a, b, l)

#         r = [mutual_info(state, a+1:a+i, b+1:b+i) for i in 1:l]
#         @assert r == result
#     end
# end



function test_binary_all_vertical_cut_ranks()
    for _ in 1:100

        m = rand(Bool, 6, 8)
        a = binary_all_vertical_cut_ranks(m)
        b = [binary_rank(m[:, 1:i]) for i in 1:8]
        @assert a == b

        m = rand(Bool, 8, 6)
        a = binary_all_vertical_cut_ranks(m)
        b = [binary_rank(m[:, 1:i]) for i in 1:6]
        @assert a == b
    end
end
test_binary_all_vertical_cut_ranks()
