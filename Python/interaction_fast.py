
import torch as pt
import numpy as np
from constants import *

# Natural constants using atomic units
permittivity = 1    # Vacuum permittivity (4 pi epsilon _0), defined as one in atomic units
q = 1               # Elementary charge, defined as one in atomic units
electron_field_strength = q / permittivity # Becomes one in atomic units

# Interaction between the two electrons
def get_e_pot(x1, x2, kappa): 
    distance = pt.abs(x1 - x2)
    return pt.exp(-kappa * distance) / distance

# Calculate electric field at origin created by electron at position r
def get_E_at_dp(x, h, kappa):
    norm = pt.sqrt(x**2 + h**2)
    x_part = x / norm
    z_part = h / norm
    f = -q * pt.exp(-kappa * norm) / norm * (1/norm + kappa)
    return  f * x_part, f * z_part

# Get the dipole moment of the dipole when the electrons are at positions r1 and r2
def get_dp_moment(x1, x2, kappa, theta_ext, E_ext, d, h):
    # Set up external electric field
    E_ext_x = E_ext * np.cos(theta_ext) * np.exp(-kappa * h)
    E_ext_z = E_ext * np.sin(theta_ext) * np.exp(-kappa * h)
    
    # Calculate electric field at the dipole
    E1x, E1z = get_E_at_dp(x1, h, kappa)
    E2x, E2z = get_E_at_dp(x2, h, kappa)
    E_at_dp_x = E_ext_x + E1x + E2x
    E_at_dp_z = E_ext_z + E1z + E2z
    E_at_dp_size = pt.sqrt(E_at_dp_x**2 + E_at_dp_z**2)
    E_at_dp_unit_x = E_at_dp_x / E_at_dp_size
    E_at_dp_unit_z = E_at_dp_z / E_at_dp_size 
    
    # Return dipole moment
    return d * E_at_dp_unit_x, d * E_at_dp_unit_z

def get_dp_pot(mx, mz, x, h, kappa): # Potential at position r due to dipole with dipole moment m
    norm = pt.sqrt(x**2 + h**2)
    return (mx * x + mz * h) / norm**3 * pt.exp(-kappa * norm)                     

# Get potential due to electron-dipole-electron interaction, for both electrons
def get_pot(x1, x2, kappa, theta_ext, E_ext, d, h):
    # Calculate potential
    mx, mz = get_dp_moment(x1, x2, kappa, theta_ext, E_ext, d, h)
    pot = get_dp_pot(mx, mz, x1, h, kappa) + get_dp_pot(mx, mz, x2, h, kappa) + get_e_pot(x1, x2, kappa)
    print(f"Minimum potential of: {pt.min(pot.flatten()).item()} \nx1: {x1.flatten()[pt.argmin(pot.flatten())].item()} \nx2: {x2.flatten()[pt.argmin(pot.flatten())].item()}")
    return pt.clamp(pt.nan_to_num(pot, nan=0.0), max=1)

def get_potential_energy(x1, x2, kappa, theta_ext, E_ext, d, h, psi_sq):
    return pt.sum(get_pot(x1, x2, kappa, theta_ext, E_ext, d, h) * psi_sq) / pt.sum(psi_sq)

def get_kinetic_energy(p1, p2, psi, psi_deriv):
    return - 0.5 * pt.sum(psi * psi_deriv) / pt.sum(psi**2)
    return pt.sum(phi * p1 ** 2) / pt.sum(phi) + pt.sum(phi * p2 ** 2) / pt.sum(phi)

