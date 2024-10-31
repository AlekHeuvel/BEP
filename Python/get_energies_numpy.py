from timeit import timeit
import torch as pt
import numpy as np
import interaction_torch as va
from constants import *
import wavefunction as wf
import scipy.optimize as opt

print(pt.cuda.is_available())
pt.device("cuda" if pt.cuda.is_available() else "cpu")
pt.set_default_tensor_type('torch.cuda.FloatTensor')
# Model parameters
theta_ext = np.pi/2
E_ext = 2.672
h = 0.625
kappa = 0
d = 1

# Define grid
x1 = pt.linspace(-dim_size, dim_size, resolution)
x2 = pt.linspace(-dim_size, dim_size, resolution)
X1, X2 = pt.meshgrid(x1, x2, indexing='ij')
 
integral_scale = resolution**2 / (4*dim_size**2)

def get_E(opt):
    # Parameters to optimize over
    global a1
    global a2
    global b
    a1, a2, b = opt[0], opt[1], opt[2]
    b = max(min(b, 10), 0.1)
    # Get wavefunctions
    phi = wf.phi_sq(X1, X2, a1, a2, b)
    psi_sq = wf.psi_sq(X1, X2, a1, a2, b)
    psi = wf.psi(X1, X2, a1, a2, b)
    psi_deriv = wf.psi_d2_dx12(X1, X2, a1, a2, b)

    # Get energies
    pot_energy = pt.sum(va.get_potential_energy(X1, X2, kappa, theta_ext, E_ext, d, h, psi_sq))  / integral_scale
    kin_energy = pt.sum(va.get_kinetic_energy(X1, X2, psi, psi_deriv)) / integral_scale
    
    total_energy = pot_energy + kin_energy
    
    # Print results
    print(f"Normalization status: {pt.sum(wf.psi_sq_normalized(X1, X2, a1, a2, b)) / integral_scale}")
    print(f"POTENTIAL ENERGY: {pot_energy}\nKINETIC ENERGY: {kin_energy}\nTOTAL ENERGY: {total_energy}") 
    print(f"a1: {a1}, a2: {a2}, b: {b}\n")
    
    return pot_energy.cpu().numpy()

# Constrain b to be positive
lin_con = {'type': 'ineq', 'fun': lambda x: x[2]-1}


a = opt.minimize(get_E, [-0.270, 0.266, 3], method='nelder-mead', options={'maxiter': 1000}, constraints=lin_con)
print(a)