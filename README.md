Julia code for simulating stabilizer states, Clifford circuits and stabilizer channels, with a focus on entanglement structure.

## Features

- Stabilizer tableau with destabilizers and phases, supporting mixed states, Clifford gates and Pauli measurements
- Efficient calculation of information theoretic quantities:Entanglement entropy, mutual information, entanglement negativity, localizable entanglement
- Samplers for random Clifford gates, and Z₂ / U(1)-symmetric Clifford gates

**preliminary implementations:**
- `SimpleClifford`: pure-state tableau for contracting stabilizer tensor networks
- `ZpSimpleClifford`: the same for generalized stabilizer states
