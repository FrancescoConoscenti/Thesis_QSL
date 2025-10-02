import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmay, sigmaz

# Staggered and Striped Magnetization
def Staggered_and_Striped_Magnetization(vstate, lattice, hi):
    ops = {}
    for i in lattice.nodes():  # just node indices
        x, y = lattice.positions[i]  # get coordinates
        staggered_sign = (-1) ** (x + y)
        ops[f"sz_{i}"] = staggered_sign * nk.operator.spin.sigmaz(hi, i)
    M_stag = sum(ops.values()) / lattice.n_nodes

    Staggered_Magnetization = vstate.expect(M_stag) 
    print(f"Staggered Magnetization = {Staggered_Magnetization.mean.real}")

    ops = {}
    for i in lattice.nodes():  # just node indices
        x, y = lattice.positions[i]  # get coordinates
        stripe_sign = (-1) ** x  # change to (-1)**y for y-stripes
        ops[f"sz_{i}"] = stripe_sign * nk.operator.spin.sigmaz(hi, i)
    M_stripe = sum(ops.values()) / lattice.n_nodes

    Striped_Magnetization = vstate.expect(M_stripe) 
    print(f"Striped Magnetization = {Striped_Magnetization.mean.real}")