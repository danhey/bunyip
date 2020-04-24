# -*- coding: utf-8 -*-
import numpy as numpy
from scipy.optimize import newton
import warnings

__all__ = ['solve_geometry', 'from_geometry']

def solve_geometry(x, y, t0=0., mask_width=0.2):

    mask = (x < mask_width / 2) & (x > -mask_width / 2)
    ph_primary = x[mask][y[mask].argmin()]
    ph_secondary = x[~mask][y[~mask].argmin()]
    dphi = ph_secondary - ph_primary
    return dphi

def from_geometry(dphi):
    psi = newton(compute_psi, 0.5, args=(dphi,))
    ecc = np.abs(ecc_func(psi))
    w = argper(ecc, psi)
    return ecc, w

def compute_psi(psi, dphi):
    return psi - np.sin(psi) - 2*np.pi*dphi 

def ecc_func(psi):
    return np.sin(0.5*(psi-np.pi))*(1.-0.5*(np.cos(0.5*(psi-np.pi)))**2)**(-0.5)

def argper(ecc, psi):
    if ecc <= 0.:
        return 0.
    return np.arccos(1./ecc * (1.-ecc**2)**0.5 * np.tan(0.5*(psi-np.pi)))