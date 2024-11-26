import torch as pt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import interaction_fast as va
import vectorized_relative as vr
import wavefunction as wf
from constants import *

print(pt.cuda.is_available())
pt.device("cuda" if pt.cuda.is_available() else "cpu")
pt.set_default_tensor_type('torch.cuda.FloatTensor')

fig = plt.figure()
ax = plt.axes(projection = "3d")

fig.subplots_adjust(left=0.25, bottom=0.4)

# ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')
# ax.set_xlabel(r'r (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
# ax.set_ylabel(r'R (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
# ax.set_zlabel(r'$ψ^2$') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
# plt.savefig("relative_wf.svg")

theta_ext = -pt.pi/2
E_ext = 3.75
h = 0.625
kappa = 0.1
d = 2

ax_theta =  fig.add_axes([0.25, 0.15, 0.65, 0.03])
ax_E_ext =  fig.add_axes([0.25, 0.10, 0.65, 0.03])
ax_h =      fig.add_axes([0.25, 0.05, 0.65, 0.03])
ax_kappa =  fig.add_axes([0.25, 0.20, 0.65, 0.03])
ax_d =      fig.add_axes([0.25, 0.25, 0.65, 0.03])

theta_slider =  Slider(ax_theta, '$θ_{ext}$ (radians)', -pt.pi, pt.pi, valinit=theta_ext)
E_ext_slider =  Slider(ax_E_ext, 'E_ext', 0, 5, valinit=E_ext)
h_slider =      Slider(ax_h, 'h', 0, 5, valinit=h)
kappa_slider =  Slider(ax_kappa, '$κ$', 0, 0.1, valinit=kappa)
d_slider =      Slider(ax_d, 'd', 0, 5, valinit=d) # Negative d means dipole points away from the electric field

# Add vertical sliders
ax_dim_size = fig.add_axes([0.1, 0.25, 0.03, 0.65])
dim_size_slider = Slider(ax_dim_size, 'dim_size', 0, 20, valinit=10, orientation='vertical')

# Initial plot
x1 = pt.linspace(-dim_size, dim_size, resolution)
x2 = pt.linspace(-dim_size, dim_size, resolution)
X1, X2 = pt.meshgrid(x1, x2, indexing='ij')
# pos = ax.imshow(va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h).cpu(), extent=[-dim_size, dim_size, -dim_size, dim_size])
# cbar = fig.colorbar(pos, ax=ax)

# Update plot
def update(val):
    # Model parameters
    theta_ext = theta_slider.val      # In radians
    E_ext = E_ext_slider.val          # In units of electron field strength, where e+ / 4 pi epsilon_0 = 1
    h = h_slider.val                  # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    kappa = kappa_slider.val          # In units of inverse bohr radius
    d = d_slider.val                  # In units of e+ a_0 = 8.4783e-30 C m
    dim_size = dim_size_slider.val    # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m

    x1 = pt.linspace(-dim_size, dim_size, resolution)
    x2 = pt.linspace(-dim_size, dim_size, resolution)
    X1, X2 = pt.meshgrid(x1, x2, indexing='ij')
    
    ax.clear()
    # ax.set_zlim(-1, 1)  
    # pot = va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h).cpu()
    # pos = ax.imshow(pot, extent=[-dim_size, dim_size, -dim_size, dim_size])

    fig.show()
    # ax.plot_surface(cp.asnumpy(X1), cp.asnumpy(X2), vanp.get_potential_diff(cp.asnumpy(X1), cp.asnumpy(X2), kappa, theta_ext, E_ext, d, h), cmap='viridis')
    # ax.plot_surface(X1, X2, va.get_potential_diff(X1, X2, kappa, theta_ext, E_ext, d, h), cmap='viridis')
    ax.plot_surface(X1.cpu(), X2.cpu(), va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h).cpu(), cmap='viridis')
    # ax.plot_surface(X1, X2, wf.phi_sq(X1, X2), cmap='viridis')
    # ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')
    ax.set_xlabel(r'x1 [$\text{a}_\text{0}$]') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    ax.set_ylabel(r'x2 [$\text{a}_\text{0}$]') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    ax.set_zlabel(r'U [$\text{E}_\text{h}$]') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
    # plt.savefig("wavefunction.svg")
    fig.canvas.draw_idle()
    

theta_slider.on_changed(update)
E_ext_slider.on_changed(update)
h_slider.on_changed(update)
kappa_slider.on_changed(update)
d_slider.on_changed(update)
dim_size_slider.on_changed(update)


plt.show()