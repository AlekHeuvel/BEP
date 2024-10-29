import torch as pt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import interaction_torch as va
import vectorized_relative as vr
import wavefunction as wf
from constants import *

a1 = 0.5
a2 = 1 
b = 1
x1 = pt.linspace(-dim_size, dim_size, resolution)
x2 = pt.linspace(-dim_size, dim_size, resolution)
X1, X2 = pt.meshgrid(x1, x2, indexing='ij')

fig = plt.figure()
ax = plt.axes(projection='3d')


fig.subplots_adjust(left=0.25, bottom=0.4)

ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')
ax.set_xlabel(r'r (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
ax.set_ylabel(r'R (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
ax.set_zlabel(r'$ψ^2$') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
plt.savefig("relative_wf.svg")
    
# ax_theta =  fig.add_axes([0.25, 0.15, 0.65, 0.03])
# ax_E_ext =  fig.add_axes([0.25, 0.10, 0.65, 0.03])
# ax_h =      fig.add_axes([0.25, 0.05, 0.65, 0.03])
# ax_kappa =  fig.add_axes([0.25, 0.20, 0.65, 0.03])
# ax_d =      fig.add_axes([0.25, 0.25, 0.65, 0.03])

# theta_slider =  Slider(ax_theta, '$θ_{ext}$ (radians)', 0.0, pt.pi, valinit=0)
# E_ext_slider =  Slider(ax_E_ext, 'E_ext', 0, 5000, valinit=0)
# h_slider =      Slider(ax_h, 'h', 0, 2, valinit=1)
# kappa_slider =  Slider(ax_kappa, '$κ$', 0, 2, valinit=1)
# d_slider =      Slider(ax_d, 'd', 0, 2, valinit=1)


# def update(val):
#     # Model parameters
#     theta_ext = theta_slider.val      # In radians
#     E_ext = E_ext_slider.val          # In units of electron field strength, where e+ / 4 pi epsilon_0 = 1
#     h = h_slider.val                  # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
#     kappa = kappa_slider.val          # In units of inverse bohr radius
#     d = d_slider.val                  # In units of e+ a_0 = 8.4783e-30 C m

#     ax.clear()
#     # ax.set_zlim(-1, 1)  
    
#     # ax.plot_surface(cp.asnumpy(X1), cp.asnumpy(X2), vanp.get_potential_diff(cp.asnumpy(X1), cp.asnumpy(X2), kappa, theta_ext, E_ext, d, h), cmap='viridis')
#     # ax.plot_surface(X1, X2, va.get_potential_diff(X1, X2, kappa, theta_ext, E_ext, d, h), cmap='viridis')
#     # ax.plot_surface(X1, X2, va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h), cmap='viridis')
#     # ax.plot_surface(X1, X2, wf.phi_sq(X1, X2), cmap='viridis')
#     ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')
#     ax.set_xlabel(r'r ($a_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
#     ax.set_ylabel(r'R ($a_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
#     ax.set_zlabel(r'$\psi^2$') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
#     # plt.savefig("wavefunction.svg")
#     fig.canvas.draw_idle()

# theta_slider.on_changed(update)
# E_ext_slider.on_changed(update)
# h_slider.on_changed(update)
# kappa_slider.on_changed(update)
# d_slider.on_changed(update)


plt.show()