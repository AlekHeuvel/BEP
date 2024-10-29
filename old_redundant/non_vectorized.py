# Pretend like everything is dirac deltas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# Natural constant and parameters
theta_0 = np.pi
E_0 = 6
d = 1
h = 1
kappa = 1
q = -1
dim_size = 4

resolution = 200


def e_e_pot(x_1, x_2): 
    print(np.ones((x_1.shape[0], x_2.shape[0])))
    return 1/(np.abs(x_1-x_2)) * np.exp(-kappa * np.abs(x_1-x_2))

def get_angle(x_1, x_2):
    y = np.sin(theta_0) * E_0 + q * h/(x_1**2 + h**2)**(3/2) + q * h/(x_2**2 + h**2)**(3/2)
    x = np.cos(theta_0) * E_0 + q * x_1/(x_1**2 + h**2)**(3/2) + q* x_2/(x_2**2 + h**2)**(3/2)
    theta = np.arctan2(y, x)
    return theta

# First exact:
def e_dp_e_pot(x_1, x_2):
        theta = get_angle(x_1, x_2)
        x_1_pot = 1 / (x_1**2 + h**2) ** (3/2) * (x_1 * np.cos(theta) + h * np.sin(theta))
        x_2_pot = 1 / (x_2**2 + h**2) ** (3/2) * (x_2 * np.cos(theta) + h * np.sin(theta))
        return x_1_pot + x_2_pot

# Very approximately
def e_dp_e_pot_approx(x_1, x_2):
    x_1_pot = x_1 + h * (np.sin(theta_0) * E_0 + (2/h**2 - 3*(x_1**2 + x_2**2)/(2*h**4))) / (np.cos(theta_0) * E_0) * (1/h**3 - 3/2 * x_1**2 / h**5)
    x_2_pot = x_2 + h * (np.sin(theta_0) * E_0 + (2/h**2 - 3*(x_1**2 + x_2**2)/(2*h**4))) / (np.cos(theta_0) * E_0) * (1/h**3 - 3/2 * x_2**2 / h**5)
    return x_1_pot + x_2_pot
    
# Better approximation?
def e_dp_e_pot_approx(x_1, x_2):
    y_on_x_term = (np.sin(theta_0) * E_0 + q * (2/h**2 - 3*(x_1**2 + x_2**2)/(2*h**4))) / (np.cos(theta_0) * E_0 )
    theta = theta_0 + (y_on_x_term - np.tan(theta_0)) / (1+np.tan(theta_0)**2) - np.tan(theta_0)*(y_on_x_term-np.tan(theta_0))**2 / (1+np.tan(theta_0)**2)**2
    x_1_pot = (x_1 * (1-theta**2/2) + h * theta) * (1/h**3 - 3/2 * x_1**2 / h**5) 
    x_2_pot = (x_2 * (1-theta**2/2) + h * theta) * (1/h**3 - 3/2 * x_2**2 / h**5) 
    return x_1_pot + x_2_pot
    

def total_exact(x_1, x_2):
     return np.minimum(np.ones((x_1.shape[0], x_2.shape[0])), d * e_e_pot(x_1, x_2) + e_dp_e_pot(x_1, x_2))

def total_approx(x_1, x_2):
     return d * e_e_pot(x_1, x_2) + e_dp_e_pot_approx(x_1, x_2)

def get_el_field(r):
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = r / r_norm
    return -q * r_unit / r_norm**2
    return -q * np.exp(-kappa * r_norm) / r_norm * (1/r_norm + kappa) * r_unit

def get_dp_moment(r_1, r_2):
    E_0_vector = np.array([E_0 * np.cos(theta_0), 0, E_0 * np.sin(theta_0)])
    E_0_screened = E_0_vector * np.exp(-kappa * h)
    E_0_screened = E_0_vector

    E_0_mesh = np.empty((3, resolution, resolution))
    E_0_mesh[0] = E_0_screened[0]
    E_0_mesh[1] = E_0_screened[1]
    E_0_mesh[2] = E_0_screened[2]
    el_field = get_el_field(r_1) + get_el_field(r_2) + E_0_mesh 
    el_field_size = np.linalg.norm(get_el_field(r_1) + get_el_field(r_2) + E_0_mesh, axis=0)
    el_field_unit = el_field / el_field_size
    return d * el_field_unit

def get_dp_pot(m, r): # Potential at position r due to dipole with dipole moment m
    r_norm = np.linalg.norm(r, axis=0)
    # mu_0 = 1/(epsilon_0 c^2) => mu_0 / 4pi = 1/(c^2) in atomic units, where c = 1/alpha in atomic units, so mu_0 / 4pi = alpha^2
    return (m*r).sum(0) / r_norm**3                                 

def get_pot(x_1, x_2):
    r_1 = np.empty((3, resolution, resolution))
    r_2 = np.empty((3, resolution, resolution))
    r_1[0] = x_1
    r_1[2] = h
    r_2[0] = x_2
    r_2[2] = h
    m = get_dp_moment(r_1, r_2)
    return get_dp_pot(m, r_1) + get_dp_pot(m, r_2) 



# Interactive plot
x_1, x_2 = np.linspace(-dim_size, dim_size, resolution), np.linspace(-dim_size, dim_size, resolution)
X_1, X_2 = np.meshgrid(x_1, x_2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('U')
ax.set_zlim(-1, 1)

fig.subplots_adjust(left=0.25, bottom=0.4)

ax_theta = fig.add_axes([0.25, 0.15, 0.65, 0.03])
ax_E_0 = fig.add_axes([0.25, 0.10, 0.65, 0.03])

theta_slider = Slider(ax_theta, 'theta_0', 0, 2*np.pi, valinit=theta_0)
E_0_slider = Slider(ax_E_0, 'E_0', 0, 1000, valinit=E_0)

def update(val):
    global theta_0
    global E_0
    
    theta_0 = theta_slider.val
    E_0 = E_0_slider.val

    ax.clear()
    ax.set_zlim(-1, 1)
    
    # ax.plot_surface(X_1, X_2, total_exact(X_1, X_2), cmap='viridis')
    ax.plot_surface(X_1, X_2, e_dp_e_pot(X_1, X_2), cmap='viridis')
    ax.plot_surface(X_1, X_2, get_pot(X_1, X_2) + 1, cmap='plasma')
    # ax.plot_surface(X_1, X_2, e_dp_e_pot_approx(X_1, X_2), cmap='plasma')
    # ax.plot_surface(X_1, X_2,  e_dp_e_pot(X_1, X_2) - e_dp_e_pot_approx(X_1, X_2))
    print(e_dp_e_pot(0, 0))
    print(e_dp_e_pot_approx(0, 0))
    fig.canvas.draw_idle()

theta_slider.on_changed(update)
E_0_slider.on_changed(update)
plt.show()