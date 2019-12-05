# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.signal import find_peaks

__all__ = ['solve_geometry', 'from_geometry']

def solve_geometry(phase, flux, period, t0=None,
                  diagnose=True):

    if t0 is None:
        t0 = phase[np.argmin(np.abs(flux - 0.))] 
        
    grad2 = np.gradient(np.gradient(flux))
    mask = ((t0 - 0.1) < phase) & ((t0 + 0.1) > phase)
    
    p = find_peaks(-grad2[mask])[0]
    heights = grad2[mask][p]
    inds_primary = p[np.argsort(heights)][:2]
    primary = np.array(sorted((phase[mask][inds_primary[0]], phase[mask][inds_primary[1]])))
    
    mask = ~mask
    p = find_peaks(-grad2[mask])[0]
    heights = grad2[mask][p]
    inds_secondary = p[np.argsort(heights)][:2]
    secondary = np.array(sorted((phase[mask][inds_secondary[0]], phase[mask][inds_secondary[1]])))
    secondary[secondary < 0] += 1.
    
    # t0s
    t0_primary = np.median(primary)
    t0_secondary = np.median(secondary)
    
    # Depths
    depth_primary = flux[(np.abs(phase - t0_primary)).argmin()]
    depth_secondary = flux[(np.abs(phase - t0_secondary)).argmin()]
    
    if diagnose:
        plt.plot(phase, (flux-flux.min()) / (flux-flux.min()).max(), linewidth=0.7,
                label='Light curve')
        # plt.plot(phase, (grad2 - grad2.min()) / (grad2 - grad2.min()).max(), linewidth=0.7,
        #         label='Second derivative')
        plt.legend()
        for pri in primary:
            plt.axvline(pri, c='red', linestyle='dashed', linewidth=0.5)
        for sec in secondary:
            plt.axvline(sec, c='blue', linestyle='dashed', linewidth=0.5)
            
    return {
        'primary_phases': primary,
        'secondary_phases': secondary,
        'primary_duration': (primary[1] - primary[0]) * period,
        'secondary_duration': (secondary[1] - secondary[0]) * period,
        't0_primary': t0_primary,
        't0_secondary': t0_secondary,
        'primary_depth': depth_primary,
        'secondary_depth': depth_secondary
    }  

def from_geometry(dphi):
    psi = newton(compute_psi, 0.5, args=(dphi,))
    ecc = np.abs(ecc_func(psi))
    w = argper(ecc, psi)
    return ecc, w

def compute_psi(psi, deltaPhi):
    return psi - np.sin(psi) - 2*np.pi*deltaPhi 

def ecc_func(psi):
    return np.sin(0.5*(psi-np.pi))*(1.-0.5*(np.cos(0.5*(psi-np.pi)))**2)**(-0.5)

def argper(ecc, psi):
    if ecc <= 0.:
        return 0.
    return np.arccos(1./ecc * (1.-ecc**2)**0.5 * np.tan(0.5*(psi-np.pi)))