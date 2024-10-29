import numpy as np
from numba import njit
from numba import vectorize, float64
import cupy as cp
from constants import resolution
from timeit import timeit
def njit(f):
    def inner(*args, **kwargs):
        val = f(*args, **kwargs)
        return val
    return inner


# Natural constants using atomic units
permittivity = 1    # Vacuum permittivity (4 pi epsilon _0), defined as one in atomic units
q = 1               # Elementary charge, defined as one in atomic units
electron_field_strength = q / permittivity # Becomes one in atomic units
alpha = 0.0072973525643 # Fine structure constant

# Other parameters
dim_size = 40   # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m

# Model parameters
theta_ext = 0
E_ext = 0
h = 1
kappa = 1
d = 1

# Redefine numpy norm
@njit
def norm(r, axis=0):
    return np.sqrt(np.sum(r*r, axis = 0))

# Define wavefunction
a1, a2 = 0.5, 1
@njit
def psi_sq(x1, x2):
    return np.exp(-2 * (x1 - a1)**2) * np.exp(-2 * (x2 - a2)**2) * (x1 - x2) ** 2

# Interaction between the two electrons
@njit
def get_e_pot(r1, r2): 
    distance = np.abs(r1 - r2)
    return np.exp(-kappa * (distance + 2*(r1-a1)**2 + 2*(r2-a2)**2)) * distance

# Calculate electric field at origin created by electron at position r
@njit
def get_E_at_dp(r):
    r_norm = norm(r, axis=0)
    r_unit = r / r_norm
    return -q * np.exp(-kappa * r_norm) / r_norm * (1/r_norm + kappa) * r_unit

# Get the dipole moment of the dipole when the electrons are at positions r1 and r2
@njit
def get_dp_moment(r1, r2):
    # Set up external electric field
    E_ext_vector = np.array([E_ext * np.cos(theta_ext), 0, E_ext * np.sin(theta_ext)])
    E_ext_screened = E_ext_vector * np.exp(-kappa * h)
    E_ext_mesh = np.empty((3, resolution, resolution))
    E_ext_mesh[0] = E_ext_screened[0]
    E_ext_mesh[1] = E_ext_screened[1]
    E_ext_mesh[2] = E_ext_screened[2]

    # Caclulate electric field at the dipole
    E_at_dp = get_E_at_dp(r1) + get_E_at_dp(r2) + E_ext_mesh 
    E_at_dp_size = norm(get_E_at_dp(r1) + get_E_at_dp(r2) + E_ext_mesh, axis=0)
    E_at_dp_unit = E_at_dp / E_at_dp_size

    # Return dipole moment
    return d * E_at_dp_unit

@njit
def get_dp_pot(m, r): # Potential at position r due to dipole with dipole moment m
    r_norm = norm(r, axis=0)
    dp_pot = (m*r).sum(axis=0) / r_norm**3
    dp_pot_screened = dp_pot * np.exp(-kappa * r_norm)
    return dp_pot_screened

@njit

# Get potential due to electron-dipole-electron interaction, for both electrons
@njit
def get_pot(x1, x2):
    # Define position vectors over the grid
    r1 = np.zeros((3, resolution, resolution))
    r2 = np.zeros((3, resolution, resolution))
    r1[0], r2[0] = x1, x2
    r1[2], r2[2] = h, h

    # Calculate potential
    m = get_dp_moment(r1, r2)
    pot = get_dp_pot(m, r1) + get_dp_pot(m, r2) + get_e_pot(x1, x2)
    return pot


def get_E():
    integral_scale = resolution**2 / (4*dim_size**2)
    normalization = np.sum(psi_sq(X1, X2)) / integral_scale
    energy = np.sum(get_pot(X1, X2) * psi_sq(X1, X2)) / normalization / integral_scale
    print(energy)


if __name__ == "__main__":
    x1, x2 = np.linspace(-dim_size, dim_size, resolution), np.linspace(-dim_size, dim_size, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    timeit(get_E, number=100)