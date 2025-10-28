import numpy as np
import os
import flax
import netket as nk



def compute_S_matrix(vstate, folder_path, hi):
    """
    Computes the S-matrix for the variational state.

    Args:
        vstate: The variational state.
        hi: The Hamiltonian instance.
    """
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    S_matrices = []

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        S_matrices.append(compute_S_matrix_single_model(vstate, hi))

    return S_matrices


def compute_S_matrix_single_model(vstate, hi):

    qgt = vstate.quantum_geometric_tensor()   # returns a QGT object (lazy or jacobian-based)
    S_matrix = qgt.to_dense()

    return S_matrix