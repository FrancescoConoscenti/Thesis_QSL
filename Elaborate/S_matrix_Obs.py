import numpy as np
import os
import flax
import netket as nk
from netket.errors import NonHolomorphicQGTOnTheFlyDenseRepresentationError
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator
import logging

logger = logging.getLogger(__name__)


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
        data = f.read()
        try:
            vstate = flax.serialization.from_bytes(vstate, data)
        except KeyError:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

    S_matrix = compute_S_matrix_single_model(vstate, hi)

    return S_matrix  # Return the last S-matrix computed


def compute_S_matrix_linear_operator(vstate):
    """
    Computes the S-matrix as a SciPy LinearOperator to completely avoid OOM errors
    from materializing the dense N_params x N_params matrix.
    """
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(vstate.parameters)
    n_params = len(flat_params)
    
    # Initialize the highly memory-efficient lazy QGT
    qgt = nk.optimizer.qgt.QGTOnTheFly(vstate, mode='complex')
    
    @jax.jit
    def qgt_matvec(v_flat):
        v_pytree = unravel_fn(v_flat)
        out_pytree = qgt @ v_pytree
        out_flat, _ = jax.flatten_util.ravel_pytree(out_pytree)
        return out_flat

    def matvec(v):
        return np.array(qgt_matvec(jnp.array(v)))

    # Because the S-matrix is Hermitian, rmatvec is equivalent to matvec
    return LinearOperator((n_params, n_params), matvec=matvec, rmatvec=matvec, dtype=np.complex128)


def compute_S_matrix_single_model(vstate, hi):

    try:
        qgt = vstate.quantum_geometric_tensor()   # returns a QGT object (lazy or jacobian-based)
        S_matrix = qgt.to_dense()
        logger.info("S-matrix computed using default vstate.quantum_geometric_tensor().to_dense()")
    except NonHolomorphicQGTOnTheFlyDenseRepresentationError:
        # Safe fallback: works for non-holomorphic ansätze
        logger.info("NonHolomorphicQGTOnTheFlyDenseRepresentationError caught. Using QGTJacobianDense fallback.")
        qgt = nk.optimizer.qgt.QGTJacobianDense(vstate, mode='complex')
        S_matrix = qgt.to_dense()

    return S_matrix