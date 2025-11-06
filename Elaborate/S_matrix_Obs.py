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
    
    # Load only the last model to compute its S-matrix
    last_model_index = number_models - 1
    with open(folder_path + f"/models/model_{last_model_index}.mpack", "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

    S_matrix = compute_S_matrix_single_model(vstate, hi)

    return S_matrix  # Return the last S-matrix computed


def compute_S_matrix_single_model(vstate, hi):

    qgt = vstate.quantum_geometric_tensor()   # returns a QGT object (lazy or jacobian-based)
    S_matrix = qgt.to_dense()

    return S_matrix