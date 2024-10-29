import numpy as np
def psi_sq(x1, x2, a1, a2, b):
    return np.exp(-2 * (x1 - a1)**2 / b) * np.exp(-2 * (x2 - a2)**2 / b) * (x1 - x2) ** 2

def psi_sq_normalized(x1, x2, a1, a2, b):
    norm = np.pi * b / 4 * (2 * (a1-a2)**2 + b)
    return psi_sq(x1, x2, a1, a2, b) / norm

def psi_sq_relative(r, R, a1, a2, b):
    return np.exp(-2/(2*b)*(r+(a1-a2))**2) * r**2 * np.exp(-4/b * (R - 1/2 * (a1 + a2))**2)

def psi_sq_relative_normalized(r, R, a1, a2, b):
    norm = 1/2 * np.sqrt(b * np.pi) * (2 * (a1-a2)**2 + b) * np.sqrt(np.pi * b) / 2
    return psi_sq_relative(r, R, a1, a2, b) / norm

def phi_sq(p1, p2, a1, a2, b):
    return ((a2-a1)**2/b + (p1-p2)**2 / 4) * np.exp(-b * p1**2 / 2 - a1**2/(2*b) -b * p2**2 / 2 - a2**2/(2*b))
