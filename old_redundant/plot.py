import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


# Model parameters
theta_0 = np.pi     # In radians
E_0 = 1             # In units of electron field strength, where e+ / 4 pi epsilon_0 = 1
h = 1               # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
kappa = 1           # In units of inverse bohr radius
d = 1               # In units of e+ a_0 = 8.4783e-30 C m

# Other parameters
dim_size = 4       # In units of bohr radius
resolution = 200


x_1, x_2 = np.linspace(-dim_size, dim_size, resolution), np.linspace(-dim_size, dim_size, resolution)
X_1, X_2 = np.meshgrid(x_1, x_2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('U')

fig.subplots_adjust(left=0.25, bottom=0.4)

ax_theta = fig.add_axes([0.25, 0.15, 0.65, 0.03])
ax_E_0 = fig.add_axes([0.25, 0.10, 0.65, 0.03])

theta_slider = Slider(ax_theta, 'θ_0 (radians)', 0, np.pi, valinit=theta_0)
E_0_slider = Slider(ax_E_0, 'E_0', 0, 1000, valinit=E_0)

def update(val):
    global theta_0
    global E_0
    
    theta_0 = theta_slider.val
    E_0 = E_0_slider.val

    ax.clear()
    ax.set_zlim(-1, 1)  
    
    ax.plot_surface(X_1, X_2, get_pot(X_1, X_2), cmap='viridis')
    fig.canvas.draw_idle()


theta_slider.on_changed(update)
E_0_slider.on_changed(update)
plt.show()
