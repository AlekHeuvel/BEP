import torch as pt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.axes import Axes
import interaction_fast as va
import vectorized_relative as vr
import wavefunction as wf
from constants import *

print(pt.cuda.is_available())
pt.device("cuda" if pt.cuda.is_available() else "cpu")
pt.set_default_tensor_type('torch.cuda.FloatTensor')

fig = plt.figure()
ax = plt.axes(projection = "3d")

# ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')
# ax.set_xlabel(r'r (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
# ax.set_ylabel(r'R (a$_0$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
# ax.set_zlabel(r'$ψ^2$') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
# plt.savefig("relative_wf.svg")

theta_ext = -pt.pi/2
E_ext = 3.5
h = 0.625
kappa = 0.1
d = 2

# Initial plot
x1 = pt.linspace(-2, 2, resolution)
x2 = pt.linspace(-2, 2, resolution)
X1, X2 = pt.meshgrid(x1, x2, indexing='ij')
# pos = ax.imshow(va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h).cpu(), extent=[-dim_size, dim_size, -dim_size, dim_size])
# cbar = fig.colorbar(pos, ax=ax)

Z = pt.load("results_physical.pt")
ax.plot_surface(X1.cpu(), X2.cpu(), Z.cpu()*1000, cmap='viridis')
# ax.plot_surface(cp.asnumpy(X1), cp.asnumpy(X2), vanp.get_potential_diff(cp.asnumpy(X1), cp.asnumpy(X2), kappa, theta_ext, E_ext, d, h), cmap='viridis')
# ax.plot_surface(X1, X2, va.get_potential_diff(X1, X2, kappa, theta_ext, E_ext, d, h), cmap='viridis')
# ax.plot_surface(X1.cpu(), X2.cpu(), va.get_pot(X1, X2, kappa, theta_ext, E_ext, d, h).cpu(), cmap='viridis')
# ax.plot_surface(X1, X2, wf.phi_sq(X1, X2), cmap='viridis')
# ax.plot_surface(X1, X2, wf.psi_sq_relative_normalized(X1, X2, a1, a2, b), cmap='viridis')


ax.set_xlabel(r'x1 [$\text{a}_\text{0}$]') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
ax.set_ylabel(r'x2 [$\text{a}_\text{0}$]') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
ax.set_zlabel(r'E [$\text{E}_\text{h} x10^{-3}$]') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
plt.show()  