using StabilizerEntanglement
using StabilizerEntanglement: flat, is_herm, double_row_sum!, erase_row!, row_auto_fill!
using Random
using Test

include("test_binary_linalg.jl")
include("test_clifford.jl")
include("test_channels.jl")

tests = [
    test_binary_all_vertical_cut_ranks,
    test1, test2, test_same_state, test_auto_fill_row,
    test_ap_neg, test_binary_random_symplectic_matrix, test_measurement,
    test_strange_mi, test_localizable_EE, test_tri_mi, test_mutual_info, test_multiple_negs,
    test_channel_decompose, test_z_dephase,
]

broken_tests = [test_depolarize_meas, test_ap_mi]

@testset "StabilizerEntanglement" begin
    @testset "$f" for f in tests
        @test (f(); true)
    end
    @testset "$f" for f in broken_tests
        @test_broken (try f(); true catch; false end)
    end
end
