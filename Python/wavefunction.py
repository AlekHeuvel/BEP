import numpy as pt
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
    norm = 1/2 * pt.sqrt(b * pt.pi) * (2 * (a1-a2)**2 + b) * pt.sqrt(pt.pi * b) / 2
    return psi_sq_relative(r, R, a1, a2, b) / norm

def phi_sq(p1, p2, a1, a2, b):
    return ((a2-a1)**2/b + (p1-p2)**2 / 4) * pt.exp(-b * p1**2 / 2 - a1**2/(2*b) -b * p2**2 / 2 - a2**2/(2*b))

def phi_sq_normalized(p1, p2, a1, a2, b):
    norm = (1+2 * (a1-a2)**2) * pt.pi / (b**2)
    return phi_sq(p1, p2, a1, a2, b) / norm

def psi_d2_dx12(x1, x2, a1, a2, b):
    exp = pt.exp(-(x1-a1)**2/b) * pt.exp(-(x2-a2)**2/b) 
    f = 4 * a1 - 6 * x1 + 2 * x2 - (4 * (x2-x1) * (x1-a1)**2) / b**2 - (4 * a2 - 6 * x2 + 2 * x1 - (4 * (x1-x2) * (x2-a2)**2) / b**2)  
    return exp * f
