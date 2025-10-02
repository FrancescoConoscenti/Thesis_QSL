import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmay, sigmaz


#Correlation function
def Corr_Struct(lattice, vstate, L, folder, hi):

    N_tot = lattice.n_nodes

    corr_r = np.zeros((L, L))
    counts = np.zeros((L, L))

    for i in range(N_tot):
        for j in range(N_tot):
            r = lattice.positions[i] - lattice.positions[j]
            corr_ij = 0.25 * (sigmaz(hi, i) * sigmaz(hi, j) + sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j))
            exp = vstate.expect(corr_ij)
            r0, r1 = int(r[0]) % L , int(r[1]) % L #PBC
            corr_r[r0, r1] += exp.mean.real
            counts[r0, r1] += 1
    corr_r /= counts 
    corr_r[0, 0] = 0  # set C(0) = 0

    plt.figure(figsize=(6,5))
    plt.imshow(corr_r, origin='lower', cmap='viridis')
    plt.colorbar(label='C(r)')
    plt.xlabel('dx')
    plt.ylabel('dy')
    plt.title('Spin-Spin Correlation Function C(r) in 2D')
    plt.xticks(np.arange(L))  # integer ticks for x-axis
    plt.yticks(np.arange(L)) 
    plt.savefig(f'{folder}/Corr.png')
    plt.show()


    #Structure factor
    # Compute the 2D Fourier transform of corr_r

    S_q = np.fft.fft2(corr_r)
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
    plt.savefig(f'{folder}/Struct.png')
    plt.show()