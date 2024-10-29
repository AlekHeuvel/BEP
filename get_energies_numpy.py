from timeit import timeit
import torch as pt
import interaction_torch as va
from constants import *
from wavefunction import phi_sq, psi_sq, psi_sq_normalized, psi_sq_relative_normalized
import scipy.optimize as opt

print(pt.cuda.is_available())

# Model parameters
theta_ext = 0
E_ext = 1
h = 1
kappa = 1
d = 5

# Define wavefunction
a1, a2 = 0.5, 1
b = 1

x1 = pt.linspace(-dim_size, dim_size, resolution)
x2 = pt.linspace(-dim_size, dim_size, resolution)
X1, X2 = pt.meshgrid(x1, x2, indexing='ij')


def get_E(opt):
    global a1
    global a2
    global b
    a1, a2, b = opt[0], opt[1], opt[2]

    integral_scale = resolution**2 / (4*dim_size**2)
    phi = phi_sq(X1, X2, a1, a2, b)
    psi = psi_sq(X1, X2, a1, a2, b)
    pot_energy = pt.sum(va.get_potential_energy(X1, X2, kappa, theta_ext, E_ext, d, h, psi))  / integral_scale
    kin_energy = pt.sum(va.get_kinetic_energy(X1, X2, phi)) / integral_scale
    total_energy = pot_energy + kin_energy
    print(pt.sum(psi_sq_relative_normalized(X1, X2, a1, a2, b)) / integral_scale)
    print(f"POTENTIAL ENERGY: {pot_energy} \nKINETIC ENERGY: {kin_energy}\nTOTAL ENERGY: {total_energy}")
    return total_energy

a = opt.minimize(get_E, [0.5, 1, 1], method='Nelder-Mead', options={'maxiter': 1000})
print(a)
