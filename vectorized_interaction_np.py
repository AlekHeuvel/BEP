
import numpy as np
from constants import *

# Natural constants using atomic units
permittivity = 1    # Vacuum permittivity (4 pi epsilon _0), defined as one in atomic units
q = 1               # Elementary charge, defined as one in atomic units
electron_field_strength = q / permittivity # Becomes one in atomic units

# Interaction between the two electrons
def get_e_pot(r1, r2, kappa): 
    distance = np.linalg.norm(r1 - r2, axis=0)
    return np.exp(-kappa * distance) / distance

# Calculate electric field at origin created by electron at position r
def get_E_at_dp(r, kappa):
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = r / r_norm
    return -q * np.exp(-kappa * r_norm) / r_norm * (1/r_norm + kappa) * r_unit

# Get the dipole moment of the dipole when the electrons are at positions r1 and r2
def get_dp_moment(r1, r2, kappa, theta_ext, E_ext, d, h):
    # Set up external electric field
    E_ext_vector = np.asarray([(E_ext * np.cos(theta_ext)), 0.0, (E_ext * np.sin(theta_ext))])
    E_ext_screened = E_ext_vector * np.exp(-kappa * h)
    E_ext_mesh = np.zeros((3, resolution, resolution))
    E_ext_mesh[0, :, :] = E_ext_screened[0]
    E_ext_mesh[1, :, :] = E_ext_screened[1]
    E_ext_mesh[2, :, :] = E_ext_screened[2]

    # Caclulate electric field at the dipole
    E_at_dp = get_E_at_dp(r1, kappa) + get_E_at_dp(r2, kappa) + E_ext_mesh 
    E_at_dp_size = np.linalg.norm(get_E_at_dp(r1, kappa) + get_E_at_dp(r2, kappa) + E_ext_mesh, axis=0)
    E_at_dp_unit = E_at_dp / E_at_dp_size

    # Return dipole moment
    return d * E_at_dp_unit

def get_dp_pot(m, r, kappa): # Potential at position r due to dipole with dipole moment m
    r_norm = np.linalg.norm(r, axis=0)
    return (m*r).sum(0) / r_norm**3 * np.exp(-kappa * r_norm)                     

# Get potential due to electron-dipole-electron interaction, for both electrons
def get_pot(x1, x2, kappa, theta_ext, E_ext, d, h):
    # Define position vectors over the grid
    r1 = np.zeros((3, resolution, resolution))
    r2 = np.zeros((3, resolution, resolution))
    r1[0, :, :] = x1
    r1[2, :, :] = h
    r2[0, :, :] = x2
    r2[2, :, :] = h

    # Calculate potential
    m = get_dp_moment(r1, r2, kappa, theta_ext, E_ext, d, h)
    pot = get_dp_pot(m, r1, kappa) + get_dp_pot(m, r2, kappa) # + get_e_pot(r1, r2, kappa)
    return np.minimum(np.nan_to_num(pot, nan=0.0), 1)

def get_pot_difference(x1, x2, kappa, theta_ext, E_ext, d, h):
    # Define position vectors on the grid
    r1 = np.zeros((3, resolution, resolution))
    r2 = np.zeros((3, resolution, resolution))
    r1[0, :, :] = x1
    r1[2, :, :] = h
    r2[0, :, :] = x2
    r2[2, :, :] = h
    
    # Getting dipole moment without influence of the electrons
    E_ext_vector = np.asarray([(E_ext * np.cos(theta_ext)), 0.0, (E_ext * np.sin(theta_ext))])
    E_ext_screened = E_ext_vector * np.exp(-kappa * h)
    E_ext_mesh = np.zeros((3, resolution, resolution))
    E_ext_mesh[0, :, :] = E_ext_screened[0]
    E_ext_mesh[1, :, :] = E_ext_screened[1]
    E_ext_mesh[2, :, :] = E_ext_screened[2]
    E_at_dp_size = np.linalg.norm(E_ext_mesh, axis=0)
    E_at_dp_unit = E_ext_mesh / E_at_dp_size
    m = d * E_at_dp_unit

    # Get potential
    pot = get_dp_pot(m, r1, kappa) + get_dp_pot(m, r2, kappa)
    return np.minimum(np.nan_to_num(pot, nan=0.0), 1)

def get_potential_energy(x1, x2, kappa, theta_ext, E_ext, d, h, psi):
    return np.sum(get_pot(x1, x2, kappa, theta_ext, E_ext, d, h) * psi) / np.sum(psi)

def get_kinetic_energy(p1, p2, phi):
    return np.sum(phi * p1 ** 2) / np.sum(phi) + np.sum(phi * p2 ** 2) / np.sum(phi)

def get_potential_diff(x1, x2, kappa, theta_ext, E_ext, d, h):
    a = get_pot(x1, x2, kappa, theta_ext, E_ext, d, h)
    b = get_pot_difference(x1, x2, kappa, theta_ext, E_ext, d, h)
    return b




