import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from constants import resolution

# Natural constants using atomic units
permittivity = 1    # Vacuum permittivity (4 pi epsilon _0), defined as one in atomic units
q = 1               # Elementary charge, defined as one in atomic units
electron_field_strength = q / permittivity # Becomes one in atomic units
alpha = 0.0072973525643 # Fine structure constant

# Interaction between the two electrons
def get_e_pot(r, kappa): 
    distance = np.linalg.norm(r, axis=0)
    return np.exp(-kappa * distance) / distance

# Calculate electric field at origin created by electron at position r
def get_E_at_dp(r, kappa):
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = r / r_norm
    return -q * np.exp(-kappa * r_norm) / r_norm * (1/r_norm + kappa) * r_unit

# Get the dipole moment of the dipole when the electrons are at positions r1 and r2
def get_dp_moment(r1, r2, kappa, theta_ext, E_ext, d, h):
    # Set up external electric field
    E_ext_vector = np.array([E_ext * np.cos(theta_ext), 0, E_ext * np.sin(theta_ext)])
    E_ext_screened = E_ext_vector * np.exp(-kappa * h)
    E_ext_mesh = np.empty((3, resolution, resolution))
    E_ext_mesh[0] = E_ext_screened[0]
    E_ext_mesh[1] = E_ext_screened[1]
    E_ext_mesh[2] = E_ext_screened[2]

    # Caclulate electric field at the dipole
    E_at_dp = get_E_at_dp(r1, kappa) + get_E_at_dp(r2, kappa) + E_ext_mesh 
    E_at_dp_size = np.linalg.norm(get_E_at_dp(r1, kappa) + get_E_at_dp(r2, kappa) + E_ext_mesh, axis=0)
    E_at_dp_unit = E_at_dp / E_at_dp_size

    # Return dipole moment
    return d * E_at_dp_unit

def get_dp_pot(m, r, kappa): # Potential at position r due to dipole with dipole moment m
    r_norm = np.linalg.norm(r, axis=0)
    # mu_0 = 1/(epsilon_0 c^2) => mu_0 / 4pi = 1/(c^2) in atomic units, where c = 1/alpha in atomic units, so mu_0 / 4pi = alpha^2
    return (m*r).sum(0) / r_norm**3 * np.exp(-kappa * r_norm)                     

# Get potential due to electron-dipole-electron interaction, for both electrons
def get_pot(r, R, kappa, theta_ext, E_ext, d, h):
    # Define position vectors over the grid
    x1 = (2 * R - r) / 2
    x2 = (2 * R + r) / 2
    r1 = np.empty((3, resolution, resolution))
    r2 = np.empty((3, resolution, resolution))
    r1[0] = x1
    r1[2] = h
    r2[0] = x2
    r2[2] = h

    # Calculate potential
    m = get_dp_moment(r1, r2, kappa, theta_ext, E_ext, d, h)
    pot = get_dp_pot(m, r1, kappa) + get_dp_pot(m, r2, kappa) + get_e_pot(r, kappa)
    return np.minimum(pot, 1)


