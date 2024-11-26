import torch as pt
import numpy as np
import interaction_fast as va
from constants import *
import wavefunction as wf

print(pt.cuda.is_available())
pt.device("cuda" if pt.cuda.is_available() else "cpu")
pt.set_default_tensor_type('torch.cuda.FloatTensor')

# Create empty arrays to store the results
results = pt.zeros((resolution, resolution))

a1 = pt.linspace(-dim_size, dim_size, resolution)
a2 = pt.linspace(-dim_size, dim_size, resolution)
A1, A2 = pt.meshgrid(a1, a2, indexing='ij')

def get_E(a1, a2):
    b = 0.55
    kappa = 0.1
    theta_ext = -pt.pi/2
    E_ext = 2
    h = 4.72
    d = 2
    
    integral_scale = resolution**2 / (4*dim_size**2)

    # x1 centered at a1
    x1 = pt.linspace(-dim_size, dim_size, resolution) + a1
    x2 = pt.linspace(-dim_size, dim_size, resolution) + a2
    X1, X2 = pt.meshgrid(x1, x2, indexing='ij')

    psi_sq = wf.psi_sq(X1, X2, a1, a2, b)
    psi = wf.psi(X1, X2, a1, a2, b)
    psi_deriv = wf.psi_d2_dx12(X1, X2, a1, a2, b)

    # Get energies
    pot_energy = pt.sum(va.get_potential_energy(X1, X2, kappa, theta_ext, E_ext, d, h, psi_sq))  / integral_scale
    kin_energy = pt.sum(va.get_kinetic_energy(X1, X2, psi, psi_deriv)) / integral_scale

    total_energy = pot_energy + kin_energy

    # Print results
    # print(f"Normalization status: {pt.sum(wf.psi_sq_normalized(X1, X2, a1, a2, b)) / integral_scale}")
    # print(f"POTENTIAL ENERGY: {pot_energy}\nKINETIC ENERGY: {kin_energy}\nTOTAL ENERGY: {total_energy}") 
    print(f"a1: {a1}, a2: {a2}, b: {b}\n")

    return total_energy.cpu()

for i in range(resolution):
    for j in range(resolution):
        results[i, j] = get_E(A1[i, j], A2[i, j])
pt.save(results, "results_physical.pt")