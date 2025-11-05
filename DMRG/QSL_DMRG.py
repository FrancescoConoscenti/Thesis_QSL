#%%
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools import hdf5_io
from numpy import linspace
from random import shuffle
import numpy as np
from tenpy.networks.site import SpinHalfSite
from DMRG.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

#%%
class J1J2Heisenberg(CouplingMPOModel):
    """A TeNPy model for the J1-J2 Heisenberg model on a square lattice."""
    def init_sites(self, model_params):
        return SpinHalfSite(conserve='Sz')

    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 4)
        Ly = model_params.get('Ly', 4)
        site = self.init_sites(model_params=model_params)
        return Square(Lx=Lx, Ly=Ly, site=site, bc='periodic', bc_MPS='finite')

    def init_terms(self, model_params):
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.0)

        # nearest neighbors (J1)
        for (u1, u2, dx) in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J1, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(1/2*J1, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            

        # next-nearest neighbors (J2)
        for (u1, u2, dx) in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(J2, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(1/2*J2, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            

#%%
if __name__ == '__main__':

    model_params = {
        'Lx' : 4,
        'Ly' : 4,
        'J1' : 1.0,
        'J2' : 0.2
    }

    model = J1J2Heisenberg(model_params=model_params)
    sites = model.lat.mps_sites()

    # Create a Neel state (alternating up/down) as the initial product state
    prod_state = ['up', 'down'] * (model.lat.N_sites // 2)
    if model.lat.N_sites % 2 == 1:
        prod_state.append('up')

    psi = MPS.from_product_state(sites, prod_state)
    dmrg_params = {
        'max_sweeps' : 10,
        'trunc_params' : {'chi_max' : 1024, 'svd_min': 1e-10},
        'chi_list' : {5:128, 10:256, 15:512, 20:1024}
        }

    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, psi = engine.run()

    # Extract energies and bond dimensions from all sweeps
    energies = engine.sweep_stats['E']
    bond_dims = engine.sweep_stats['max_chi']
    sweeps = range(len(energies))
    energies_per_site = [E / model.lat.N_sites for E in energies]

    plot_DMRG_energies(energies_per_site, bond_dims, sweeps, model_params)    

    Correlations_Structure_Factor(psi, model_params, model)