import numpy as np
import torch as pt

def psi(x1, x2, a1, a2, b):
    return pt.exp(- (x1 - a1)**2 / b) * pt.exp(- (x2 - a2)**2 / b) * (x1 - x2)

def psi_sq(x1, x2, a1, a2, b):
    return pt.exp(-2 * (x1 - a1)**2 / b) * pt.exp(-2 * (x2 - a2)**2 / b) * (x1 - x2) ** 2

def psi_sq_normalized(x1, x2, a1, a2, b):
    norm = pt.pi * b / 4 * (2 * (a1-a2)**2 + b)
    return psi_sq(x1, x2, a1, a2, b) / norm

def psi_sq_relative(r, R, a1, a2, b):
    return pt.exp(-2/(2*b)*(r+(a1-a2))**2) * r**2 * pt.exp(-4/b * (R - 1/2 * (a1 + a2))**2)

def psi_sq_relative_normalized(r, R, a1, a2, b):
    norm = 1/2 * np.sqrt(b * pt.pi) * (2 * (a1-a2)**2 + b) * np.sqrt(pt.pi * b) / 2
    return psi_sq_relative(r, R, a1, a2, b) / norm

def phi_sq(p1, p2, a1, a2, b):
    return ((a2-a1)**2/b + (p1-p2)**2 / 4) * pt.exp(-b * p1**2 / 2 - a1**2/(2*b) -b * p2**2 / 2 - a2**2/(2*b))

def phi_sq_normalized(p1, p2, a1, a2, b):
    norm = (1+2 * (a1-a2)**2) * pt.pi / (b**2)
    return phi_sq(p1, p2, a1, a2, b) / norm

def psi_d2_dx12(x1, x2, a1, a2, b):
    exp = pt.exp(-(x1-a1)**2/b) * pt.exp(-(x2-a2)**2/b) 
    # f = 4*a1 - 6*x1 + 2*x2 - (4*(x2-x1)*(x1-a1)**2) / b**2 - (4*a2 - 6*x2 + 2*x1 - (4*(x1-x2)*(x2-a2)**2) / b**2)  
    f1 =  1/b * (-6*x1  + 4*x1**3/b - 8*a1*x1**2/b + 4*a1**2*x1/b + 4*a1 + 2*x2 - 4*x1**2*x2/b + 8*x1*x2*a1/b - 4*x2*a1**2/b)
    f2 = -1/b * (-6*x2  + 4*x2**3/b - 8*a2*x2**2/b + 4*a2**2*x2/b + 4*a2 + 2*x1 - 4*x2**2*x1/b + 8*x2*x1*a2/b - 4*x1*a2**2/b)
    
    f1 =  (4 * x1**3 + (-4 * x2 - 8 * a1) * x1**2 + (8 * a1 * x2 - 6 * b + 4 * a1**2) * x1 + (2 * b - 4 * a1**2) * x2 + 4 * a1 * b) / b**2
    f2 = -(4 * x2**3 + (-4 * x1 - 8 * a2) * x2**2 + (8 * a2 * x1 - 6 * b + 4 * a2**2) * x2 + (2 * b - 4 * a2**2) * x1 + 4 * a2 * b) / b**2
    return exp * (f1 + f2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from constants import *    

    print(pt.cuda.is_available())
    pt.device("cuda" if pt.cuda.is_available() else "cpu")
    pt.set_default_tensor_type('torch.cuda.FloatTensor')

    fig = plt.figure()
    ax = plt.axes(projection = "3d")

    fig.subplots_adjust(left=0.25, bottom=0.4)

    X1 = pt.linspace(-3, 3, resolution)
    X2 = pt.linspace(-3, 3, resolution)
    X1, X2 = pt.meshgrid(X1, X2, indexing='ij')

    ax.plot_surface(X1.cpu(), X2.cpu(), (psi_sq_normalized(X1, X2, 3.209e-04, -5.885e-05, 0.55)).cpu(), cmap='viridis')
    
    # Set z limits to be between 0 and 1
    # ax.set_zlim(-0.01, 0.01)  
    # ax.plot_surface(X1.cpu(), X2.cpu(), psi_d2_dx12(X1, X2, -10, -0.15, 10).cpu(), cmap='spring')
    
    ax.set_xlabel(r'$x1$ (a$_{0}$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    ax.set_ylabel(r'$x2$ (a$_{0}$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    ax.set_zlabel(r'$ψ^2$ (a$_{0}^{-2}$)') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
    
    # ax.set_xlabel(r'$p1$ ($\hbar$ a$_{0}^{-1}$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    # ax.set_ylabel(r'$p2$ ($\hbar$ a$_{0}^{-1}$)') # In units of bohr radius = 4 pi epsilon_0 h_bar^2 / (m_e e+^2), = 5.2917e-11 m
    # ax.set_zlabel(r'$φ^2$ (a$_{0}^{2}$ $\hbar^{-2}$)') # In units of hartree energy, with E_h = e^2 / (4 pi epsilon_0 a_0) = 4.3597e-18 J
    plt.show()