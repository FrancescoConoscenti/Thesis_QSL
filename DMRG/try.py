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
from tenpy.algorithms.exact_diag import get_full_wavefunction
from DMRG.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor
from DMRG.importance_sampling import *
from DMRG.Fidelities import *
from DMRG.QSL_DMRG import *


# Usage
model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.0
    }
N_sites = model_params['Lx'] **2

hamiltonian = J1J2Heisenberg(model_params=model_params)
DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params)

samples, weights = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, complex_amplitude=False)
    
print(f"Sampled {len(samples)} unique configurations")