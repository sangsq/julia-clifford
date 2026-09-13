module StabilizerEntanglement

using LinearAlgebra
using Random
using Statistics

export binary_uppertrianglize, binary_uppertrianglize!, binary_partial_gaussian!,
    binary_row_echlon, binary_row_echlon!, binary_null_space, binary_rank,
    binary_bidirectional_gaussian, binary_bidirectional_gaussian!, binary_inner,
    binary_all_diagonal_ranks, binary_all_vertical_cut_ranks, binary_all_vertical_cut_ranks!,
    binary_symplectic_inner, binary_random_symplectic_matrix, binary_random_orthogonal_matrix,
    binary_jordan_wigner_transform!, binary_charge_conserving_symplectic_mat,
    binary_random_sign_free_symplectic_matrix

export StabState, Clifford, PauliString,
    all_up, all_plus, white_state, epr_pairs, random_state,
    random_clifford, random_cc_clifford, random_z2_clifford, abs_cliff_trace,
    clifford_action!, fps_measurement!, measurement!,
    entropy, left_ee_on_all_cuts, right_ee_on_all_cuts, mutual_info, mutual_neg,
    multiple_negs, ap_negativity, ap_mutual_info, strange_AB_mi, localizable_EE, tri_mi

export depolarize!, dephase_z!, replace_up!

export StabChannel, identity_channel, clifford_action_on_channel!, z_dephase!, z_damp!,
    add_qubits!, channel_decompose

export Z2_rip, cqca_g, cqca_f, ddet, cqca_to_bmat, sample_cqca, get_all_cqca

include("binary_linalg.jl")
include("clifford.jl")
include("channels.jl")
include("choi_channels.jl")
include("z2_ri_poly.jl")

module SimpleClifford

using ..StabilizerEntanglement: binary_all_vertical_cut_ranks, binary_bidirectional_gaussian!,
    binary_random_sign_free_symplectic_matrix, binary_random_symplectic_matrix

export StabState, Clifford, nsites, tproduct, all_up, all_plus, epr_pairs, random_state,
    random_clifford, random_css_clifford, clifford_action!, measure_out!, measure_out,
    contract_sites!, contract_sites, reorder_sites, ee_on_all_cuts, mutual_info

include("simple_clifford.jl")

end

module ZpSimpleClifford

using ..StabilizerEntanglement: binary_bidirectional_gaussian!

export Zp, zp_uppertrianglize, zp_uppertrianglize!, zp_rank, zp_bidirectional_gaussian,
    zp_bidirectional_gaussian!, zp_all_vertical_cut_ranks, zp_all_vertical_cut_ranks!,
    zp_symplectic_inner, zp_random_symplectic_matrix, zp_random_sign_free_symplectic_matrix

export StabState, Clifford, check_xz, nsites, tproduct, all_up, cconj, epr_pairs, random_state,
    random_clifford, random_css_clifford, clifford_action!, measure_out!, measure_out,
    contract_sites!, contract_sites, reorder_sites, ee_on_all_cuts

include("zp_alg.jl")
include("zp_simple_clifford.jl")

end

end
