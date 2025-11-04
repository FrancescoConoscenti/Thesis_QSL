import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmay, sigmaz


#Correlation function
def Corr_Struct(lattice, ket_gs, L, hi, folder=None, plot=False):

    N_tot = lattice.n_nodes

    corr_r = np.zeros((L, L))
    counts = np.zeros((L, L))

    for i in range(N_tot):
        for j in range(N_tot):
            r = lattice.positions[i] - lattice.positions[j]
            op = 0.25 * (sigmaz(hi, i) @ sigmaz(hi, j) + sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
            # Calculate exact expectation value: <psi|O|psi>
            exp_val = np.vdot(ket_gs, op @ ket_gs)
            r0, r1 = int(r[0]) % L , int(r[1]) % L #PBC
            corr_r[r0, r1] += exp_val.real
            counts[r0, r1] += 1
    corr_r /= counts 
    corr_r[0, 0] = 0  # set C(0) = 0
    
    #Structure factor
    # Compute the 2D Fourier transform of corr_r
    S_q = np.fft.fft2(corr_r)

    if plot:
        if folder is None:
            raise ValueError("A folder must be provided if plot is True.")
        plt.figure(figsize=(6,5))
        plt.imshow(corr_r, origin='lower', cmap='viridis')
        plt.colorbar(label='C(r)')
        plt.xlabel('dx')
        plt.ylabel('dy')
        plt.title('Spin-Spin Correlation Function C(r) in 2D')
        plt.xticks(np.arange(L))  # integer ticks for x-axis
        plt.yticks(np.arange(L)) 
        plt.savefig(f'{folder}/physical_obs/Corr.png')
        plt.show()

        S_q_periodic = np.zeros((L+1, L+1), dtype=S_q.dtype)
        S_q_periodic[:L, :L] = S_q  
        S_q_periodic[L, :] = S_q_periodic[0, :]    
        S_q_periodic[:, L] = S_q_periodic[:, 0]    

        plt.figure(figsize=(6,5))
        plt.imshow(np.abs(S_q_periodic), origin='lower', cmap='viridis')
        plt.colorbar(label='|S(q)|')
        plt.xlabel('q_x')
        plt.ylabel('q_y')
        plt.title('Structure Factor S(q)')
        plt.xticks([0, 1/2*L, L], ['0', 'π', '2π'])
        plt.yticks([0, 1/2*L, L], ['0', 'π', '2π'])
        plt.savefig(f'{folder}/physical_obs/Struct.png')
        plt.show()

    return S_q

def Corr_Struct_exact(lattice, ket_gs, L, hi):
    """
    Computes the spin-spin correlation C(r) and its Fourier transform,
    the static structure factor S(q), from an exact ground state vector.

    Args:
        lattice: The NetKet graph object.
        ket_gs (np.ndarray): The ground state vector.
        L (int): The linear size of the lattice.
        hi: The Hilbert space.

    Returns:
        S_q (np.ndarray): The 2D static structure factor.
    """
    N_tot = lattice.n_nodes
    corr_r = np.zeros((L, L))
    counts = np.zeros((L, L))

    for i in range(N_tot):
        for j in range(N_tot):
            r = lattice.positions[i] - lattice.positions[j]
            op = 0.25 * (sigmaz(hi, i) @ sigmaz(hi, j) + sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
            exp_val = np.vdot(ket_gs, op @ ket_gs)
            r0, r1 = int(r[0]) % L, int(r[1]) % L
            corr_r[r0, r1] += exp_val.real
            counts[r0, r1] += 1
    corr_r /= counts
    corr_r[0, 0] = 0
    return np.fft.fft2(corr_r)

def Dimer_Corr_Struct_exact(lattice, ket_gs, L, hi):
    """
    Computes the dimer-dimer correlation C_d(r) and its Fourier transform,
    the dimer structure factor S_d(q), from an exact ground state vector.

    The dimer operator is defined as D_i = S_i . S_{i+x}.

    Args:
        lattice: The NetKet graph object.
        ket_gs (np.ndarray): The ground state vector.
        L (int): The linear size of the lattice.
        hi: The Hilbert space.

    Returns:
        S_d_q (np.ndarray): The 2D dimer structure factor.
    """
    N_tot = lattice.n_nodes
    corr_d_r = np.zeros((L, L), dtype=np.complex128)
    counts = np.zeros((L, L), dtype=int)

    # Helper to create the dimer operator D_i = S_i . S_{i+x}
    def get_dimer_op(site_i):
        pos_i = lattice.positions[site_i]
        # Find neighbor in +x direction, handling PBC
        pos_neighbor = (pos_i + np.array([1, 0])) % L
        site_j = np.where((lattice.positions == pos_neighbor).all(axis=1))[0][0]
        
        op = 0.25 * (sigmaz(hi, site_i) @ sigmaz(hi, site_j) +
                     sigmax(hi, site_i) @ sigmax(hi, site_j) +
                     sigmay(hi, site_i) @ sigmay(hi, site_j))
        return op

    # Pre-calculate all dimer operators and their expectation values <D_i>
    dimer_ops = [get_dimer_op(i) for i in range(N_tot)]
    dimer_expects = np.array([np.vdot(ket_gs, op @ ket_gs) for op in dimer_ops])

    # Calculate the connected correlation function <D_i D_j> - <D_i><D_j>
    for i in range(N_tot):
        for j in range(N_tot):
            # <D_i D_j>
            op_ij = dimer_ops[i] @ dimer_ops[j]
            exp_val_ij = np.vdot(ket_gs, op_ij @ ket_gs)

            # C_d(i,j) = <D_i D_j> - <D_i><D_j>
            corr = exp_val_ij - dimer_expects[i] * dimer_expects[j]

            # Average over translations
            r = lattice.positions[i] - lattice.positions[j]
            r0, r1 = int(r[0]) % L, int(r[1]) % L
            corr_d_r[r0, r1] += corr
            counts[r0, r1] += 1

    corr_d_r /= counts

    return np.fft.fft2(corr_d_r)