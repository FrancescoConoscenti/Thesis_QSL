# --- Calculate and plot correlations and structure factor ---
import numpy as np
from DMRG.plot.Plotting import *

def Correlations_Structure_Factor(psi, model_params, model):    
    Sz_Sz = psi.correlation_function("Sz", "Sz")
    Sp_Sm = psi.correlation_function("Sp", "Sm")

    Lx = model_params['Lx'] 
    Ly = model_params['Ly']
    corr_r = np.zeros((Lx, Ly), dtype=np.complex128)
    counts = np.zeros((Lx, Ly), dtype=int)

    for i in range(model.lat.N_sites):
        for j in range(model.lat.N_sites):
            xi, yi, _ = model.lat.mps2lat_idx(i) # Unpack x, y, and ignore the third component (unit cell index)
            xj, yj, _ = model.lat.mps2lat_idx(j) # Unpack x, y, and ignore the third component (unit cell index)
            dx, dy = (xi - xj) % Lx, (yi - yj) % Ly
            corr_r[dx, dy] += Sz_Sz[i, j] + 0.5 * (Sp_Sm[i, j] + np.conj(Sp_Sm[j, i]))
            counts[dx, dy] += 1
    
    corr_r /= counts
    S_q = np.fft.fft2(corr_r)
    S_q_periodic = np.zeros((Lx+1, Ly+1), dtype=S_q.dtype)
    S_q_periodic[:Lx, :Ly] = S_q
    S_q_periodic[Lx, :] = S_q_periodic[0, :]
    S_q_periodic[:, Ly] = S_q_periodic[:, 0]


    plot_correlation_function(corr_r, model_params)
    plot_structure_factor(S_q_periodic, model_params)


def calculate_R(S_q, L):
    Lx, Ly = S_q.shape
    S_q_copy = np.abs(S_q.copy())
    S_q_copy[0, 0] = 0.0
    peak_idx = np.unravel_index(np.argmax(S_q_copy), S_q.shape)
    Q_val = S_q_copy[peak_idx]

    if Q_val == 0:
        return 0.0

    neighbor_x_idx = ((peak_idx[0] + 1) % Lx, peak_idx[1])
    neighbor_y_idx = (peak_idx[0], (peak_idx[1] + 1) % Ly)

    S_Q_plus_dq_x = S_q_copy[neighbor_x_idx]
    S_Q_plus_dq_y = S_q_copy[neighbor_y_idx]

    R_x = 1.0 - (S_Q_plus_dq_x / Q_val)
    R_y = 1.0 - (S_Q_plus_dq_y / Q_val)
    R_avg = 0.5 * (R_x + R_y)
    return R_avg