import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmay, sigmaz
from scipy.sparse.linalg import eigsh
from jax import numpy as jnp

#Exact gs
def Exact_gs(L, J2, ha, J1J2, spin=True):

    if J1J2==True and spin==True:
        E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
        #E_gs, ket_gs = nk.exact.full_ed(ha, compute_eigenvectors=True)

    if J1J2==True and spin==False:
        graph_scaled = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
        hi_scaled = nk.hilbert.Spin(s=0.5, N=graph_scaled.n_nodes)
        H = nk.operator.Heisenberg(hilbert=hi_scaled, graph=graph_scaled, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()
        H_sparse = H.to_sparse(jax_=False).tocsc()
        E_gs, vecs = eigsh(H_sparse, k=1, which="SA")

    if J1J2==False and spin==True:
        graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        hi_scaled = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)#, total_sz=0) 
        H = nk.operator.Ising(hilbert=hi_scaled, graph=graph, h=jnp.float64(2), J=1.0)
        H_sparse = H.to_sparse(jax_=False).tocsc()
        E_gs, vecs = eigsh(H_sparse, k=1, which="SA")
        ket_gs = vecs[:,0]

    if J1J2==False and spin==False:
        graph_scaled = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        hi_scaled = nk.hilbert.Spin(s=0.5, N=graph_scaled.n_nodes)
        H = nk.operator.Ising(hilbert=hi_scaled, graph=graph_scaled, h=jnp.float64(J2), J=1.0)
        H_sparse = H.to_sparse(jax_=False).tocsc()
        E_gs, vecs = eigsh(H_sparse, k=1, which="SA")


    E_exact = E_gs[0]/(L*L*4)
    print(f"Exact ground state energy per site= {E_exact}")

    if spin==True:
        return E_exact, ket_gs
    else:
        return E_exact



#Relative Error
def Relative_Error(E_vs, E_exact, L):
    e = np.abs((E_vs - E_exact))/(L*L)
    return e

#Total magnetization on Z
def Magnetization(vstate, lattice, hi):
    tot_magn = sum([sigmaz(hi, i) for i in lattice.nodes()])
    tot_magn_vstate = vstate.expect(tot_magn).mean.real
    print(f"Magnetization = {tot_magn_vstate}" )

#Variance
def Variance(log, folder=None):
    variance_history = log.data["Energy"]["Variance"]
    variance = variance_history[-1].real
    print(f"Variance = {variance}")

    if folder is not None:
        plt.figure()
        plt.plot(variance_history.real)
        plt.title("Variance vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Variance")
        plt.yscale("log")
        plt.savefig(f'{folder}/Variance_log.png')
        plt.close()

    return variance

#Vscore
def Vscore(L, variance, E_vs):
    v_score = L*L*variance/(E_vs*L*L*4)
    print(f"V-score = {v_score}")