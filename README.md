# StabilizerEntanglement.jl

[![CI](https://github.com/sangsq/julia-clifford/actions/workflows/CI.yml/badge.svg)](https://github.com/sangsq/julia-clifford/actions/workflows/CI.yml)

Julia code for simulating stabilizer states, Clifford circuits and stabilizer channels, with a focus on entanglement structure.

## Features

- Stabilizer tableau with destabilizers and phases, supporting mixed states, Clifford gates and Pauli measurements
- Entanglement entropy on all bipartition cuts, mutual information, entanglement negativity, tripartite mutual information, localizable entanglement
- Samplers for random Clifford gates, including Z₂-symmetric and charge-conserving ensembles
- Stabilizer channels represented by vectorized Choi states, with decomposition into discarded, dephased and identity parts
- Clifford quantum cellular automata over Z₂ Laurent polynomials
- `SimpleClifford`: sign-free pure-state tableau for contracting stabilizer tensor networks
- `ZpSimpleClifford`: the same for qudits of prime dimension p

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/sangsq/julia-clifford")
```

## Usage

```julia
using StabilizerEntanglement

state = all_up(8)
for t in 1:20, i in 1:7
    clifford_action!(random_clifford(2), state, [i, i+1])
end
measurement!(state, (0, Bool[0, 1]), [4])

left_ee_on_all_cuts(state)
mutual_info(state, 1:3, 6:8)
mutual_neg(state, 1:3, 6:8)

ch = identity_channel(4)
depolarize!(ch, 1)
channel_decompose(ch)
```

```julia
using StabilizerEntanglement.ZpSimpleClifford

state = measure_out!(random_state(Zp{3}, 6), [1])
ee_on_all_cuts(state)
```

## Layout

- `src/`: package source; `src/old_implementations/` keeps earlier versions that are no longer loaded
- `test/`: run with `Pkg.test()`
- `notebooks/`: research notebooks; they predate the package and `include` source files directly
