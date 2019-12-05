# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .knn import KNN

import numpy as np
import ellc
from scipy import optimize
import lightkurve as lk
import matplotlib.pyplot as plt

__all__ = ["Bunyip"]

class Bunyip:
    def __init__(self, phase, flux, flux_err=None,
                shape_1='roche', shape_2='roche', period=None):
        """A tool for fitting eclipsing binary light curves
        
        Parameters
        ----------
        time : np.ndarray
            Time values
        flux : np.ndarray
            Flux values
        period : float, optional
            Period of the system, by default None
        flux_err : np.ndarray, optional
            Errors associated with flux, by default None
        """
        self.phase = phase
        self.flux = flux / np.median(flux)
        if flux_err is None:
            flux_err = np.zeros_like(flux)
        self.flux_err = flux_err

        # Initialise network
        self.model = self.initialize_network()
        self.period = period
        # Initialise fit parameters
        self.parameters = {
            'q':    1.,
            'rsum': 0.5,
            'rratio': 0.5,
            'fc': 0,
            'fs': 0,
            'sbratio': 1.,
            'incl': 89.,
            't0': 0.,#self.phase[np.argmin(self.flux)],
            'mean': 0.,#np.mean(self.flux) - 1.,
            'log_f': np.std(self.flux),
            # 'heat_1': 0.,
            # 'heat_2': 0.,
            'gdc_1': 0,
            'gdc_2': 0,
            # 'ldc_1': 0.5,
            # 'ldc_2': 0.5,
        }

        self.ellc_parameters = {
            'shape_1': shape_1,
            'shape_2': shape_2
        }
        
        self.nll = lambda *args: -self.lnlike(*args)
        self.nll_wrapper = lambda *args: -self.lnlike_wrapper(*args)

    @staticmethod
    def from_lightcurve(time, flux, flux_err=None, period=None, t0=0., bins=None):
        """Creates a Bunyip object from an unphased light curve where the 
        period is known.
        
        Parameters
        ----------
        time : np.ndarray
            Time values
        flux : np.ndarray
            Flux values
        flux_err : np.ndarray, optional
            Flux error values, by default None
        period : float, optional
            Orbital period, by default None
        t0 : float, optional
            Phase of superior conjunction, by default 0.
        
        Returns
        -------
        Bunyip
            [description]
        
        Raises
        ------
        ValueError
            [description]
        """
        if period is None:
            raise ValueError('I really should calculate the period')

        phase = (t0 % period) / period    
        fold_time = (((time - phase * period) / period) % 1)
        fold_time[fold_time > 0.5] -= 1
        sorted_args = np.argsort(fold_time)

        if flux_err is None:
            flux_err = np.zeros_like(time)

        return Bunyip(fold_time[sorted_args], 
                        flux[sorted_args], 
                        flux_err[sorted_args],
                        period=period)

    def lc_model(self, time=None):
            
        rsum = self.parameters['rsum']
        rratio = self.parameters['rratio']
        r1, r2 = rsum/(1.+rratio), rsum*rratio/(1.+rratio)

        lc = ellc.lc(self.phase,
                     t_zero=self.parameters['t0'],
                     q=self.parameters['q'],
                     period=1., # Period is always 1 since we fit in phase space
                     radius_1=r1, 
                     radius_2=r2,
                     incl=self.parameters['incl'],
                     sbratio=self.parameters['sbratio'], 
                     f_c=self.parameters['fc'], 
                     f_s=self.parameters['fs'],
                     gdc_1=self.parameters['gdc_1'],
                     gdc_2=self.parameters['gdc_2'],
                     shape_1=self.ellc_parameters['shape_1'], 
                     shape_2=self.ellc_parameters['shape_1'],
                    #  heat_1=self.parameters['heat_1'],
                    #  heat_2=self.parameters['heat_2'],
                    #  ld_1='lin',
                    #  ld_2='lin',
                    #  ldc_1=self.parameters['ldc_1'],
                    #  ldc_2=self.parameters['ldc_2']
                    )

        return lc + self.parameters['mean']
    
    def initialize_network(self, model_path=None):
        """ Move this to a class please"""
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.initializers import glorot_uniform
            from tensorflow.keras.utils import CustomObjectScope
        except:
            raise ImportError("You need TensorFlow for this")

        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__),  "network/RELU_2000_2000_lr=1e-05_norm_insert2000layer-1571628331/NN.h5")
        
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model(model_path)
        
        return model
        
    def predict(self, binned_flux, model=None, norm=True):
        if model is None:
            model = self.model
        y_hat = model.predict(binned_flux[None,:])
            
        if norm:
            mu= np.array([
                0.6205747949371053,
                0.2374090928468623,
                0.1891195153173617,
                1.3006089700783283,
                69.9643427508551,
                0.17749621516829056,
                179.6479435131075,
            ])
            std = np.array([
                0.22820790194476795,
                0.08166430725337233,
                0.05891981424090313,
                0.4059874833585892,
                11.465339377838976,
                0.12821797216376407,
                103.59690197983575,
            ])
            y_hat = y_hat * std + mu
        return y_hat
   
    def update_from_geometry(self, diagnose=False):
        from .geometry import solve_geometry, from_geometry
        geometry_vals = solve_geometry(self.phase, self.flux, self.period, t0=self.parameters['t0'], diagnose=diagnose)
        dphi = np.abs(geometry_vals['t0_primary'] - geometry_vals['t0_secondary'])
        ecc, per0 = from_geometry(dphi)
        self.parameters.update({
            'fc': np.sqrt(ecc) * np.cos(per0),
            'fs': np.sqrt(ecc) * np.sin(per0),
            # 'sbratio': geometry_vals['secondary_depth']/geometry_vals['primary_depth']
        })

    def update_from_knn(self):
        model = KNN()
        lc = lk.LightCurve(self.phase, self.flux)
        binned = lc.bin(bins=101).normalize()
        # plt.plot(binned.flux)
        prediction = model.predict(binned.flux)[0]
        # db_param_names = ['rsum', 'rratio', 'eccen', 'omega', 'incl', 'q', 'T1', 'tratio']
        # rsum, rratio, ecc, per0, incl, q, t1, tratio = prediction
        q, incl, rsum, rratio, t1, tratio, ecc, per0 = prediction
        self.parameters.update({
            'q': q,
            'rsum': rsum,
            'rratio': rratio,
            'fc': np.sqrt(ecc)*np.cos(np.radians(per0)),
            'fs': np.sqrt(ecc)*np.sin(np.radians(per0)),
            'sbratio': tratio,
            'incl': incl
        })
        # self.update_parameters(prediction)

    def update_from_network(self, **kwargs):
        lc = lk.LightCurve(self.phase, self.flux)
        binned = lc.bin(bins=100).normalize()
        prediction = self.predict(binned.flux, **kwargs)[0]
        self.update_parameters(prediction)
    
    def update_parameters(self, prediction):
        old_params = self.parameters.copy()
        q, r1, r2, tratio, incl, ecc, per0 = prediction
        self.parameters.update({
            'q': q,
            'rsum': r1+r2,
            'rratio': r2/r1,
            'fc': np.sqrt(ecc)*np.cos(np.radians(per0)),
            'fs': np.sqrt(ecc)*np.sin(np.radians(per0)),
            'sbratio': tratio,
            'incl': incl
        })

        if not np.isfinite(self.lnprior()):
            print(self.parameters)
            self.parameters = old_params
            print('The network failed to find a solution, defaulting to original values. Or grid search here?')

    def lnprior(self):
        rsum = self.parameters['rsum']
        rratio = self.parameters['rratio']
        r1, r2 = rsum/(1.+rratio), rsum*rratio/(1.+rratio)
        
        ecc = self.parameters['fc']**2 + self.parameters['fs']**2
        per0 = np.arctan2(self.parameters['fs'],self.parameters['fc'])
        if ((0 < self.parameters['q']) & (self.parameters['incl'] <= 90) & (0 < self.parameters['sbratio']) & 
            ((-0.5) <= self.parameters['t0'] <= (0.5)) & (0 < r1 < 1) & (0 < r2 < 1) & (0 <= ecc < 1)):
            return 0.
        else:
            return -np.inf
        
    def lnlike_wrapper(self, params, *vars):
        return self.lnlike(params, vars)
    
    def lnlike(self, params, vars):
        """The ln likelihood of the model.
        
        Parameters
        ----------
        params : list
            List of parameter values corresponding to the var names
        vars : list
            List of parameter names corresponding to the params
        
        Returns
        -------
        float
            Value of the ln likelihood at the given values
        """
        for param, var in zip(params, vars):
            self.parameters.update({var : param})
            
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        model_flux = self.lc_model()
        sigma2 = self.flux_err ** 2 + model_flux ** 2 * np.exp(2 * self.parameters['log_f'])
        try:
            ln_lc = -0.5*np.sum((self.flux-model_flux)**2/sigma2 + np.log(sigma2))
        except:
            return -np.inf
        if np.any(np.isnan(ln_lc)):
            return -np.inf
        return ln_lc
    
    def optimize(self, vars=None, **kwargs):
        """Optimises the `parameters` of the Bunyip object with Scipy
        
        Parameters
        ----------
        vars : list, optional
            List of vars to optimize, by default None
        
        Returns
        -------
        dict
            Results of the optimization
        """
        if vars is None:
            vars = list(self.parameters.keys())

        x0 = [self.parameters[var] for var in vars]   
        soln = optimize.minimize(self.nll, x0, args=(vars), **kwargs)
        for var, val in zip(vars, soln.x):
            self.parameters.update({var: val})
        return soln
    
    def differential_evolution(self, bounds):
        vars = list(self.parameters.keys())

        if not len(vars) == len(bounds):
            raise ValueError("You must have as many bounds as parameters for \
                            differential evolution")
        
        soln = optimize.differential_evolution(self.nll_wrapper, bounds, args=(vars))
        for val, key in zip(soln.x, vars):
            self.parameters.update({key:val})
        return soln
        
    def plot_model(self, ax=None):
        model_flux = self.lc_model()
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.phase, model_flux, label='Model')
        ax.plot(self.phase, self.flux, '.k', label='Data')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        plt.legend()
        return ax
    
    def run_emcee(self, burnin=1000, draws=2000, nwalkers=32, **kwargs):
        import emcee
        vars = list(self.parameters.keys())
        init = list(self.parameters.values())
        pos = init + 1e-4*np.random.randn(nwalkers, len(init))
        ndim = len(vars)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnlike_wrapper, 
                                        args=(vars),
                                        moves=[(emcee.moves.DEMove(), 0.8), 
                                        (emcee.moves.DESnookerMove(), 0.2),],)
        sampler.run_mcmc(pos, burnin+draws, progress=True);
        samples = sampler.get_chain(flat=True, discard=burnin)
        self.sampler = sampler
        return samples

    def corner_plot(self, trace, **kwargs):
        import corner
        ax = corner.corner(trace, labels=list(self.parameters.keys()),
                    show_titles=True, **kwargs);
        return ax

    def plot_samples_from_trace(self, trace, n=100, ax=None, **kwargs):
        vars = list(self.parameters.keys())
        samples = trace[np.random.choice(len(trace), size=n)]

        if ax is None:
            fig, ax = plt.subplots()
        for i in samples:
            [self.parameters.update({key:val}) for key, val in zip(vars, i)]
            model_flux = self.lc_model()
            plt.plot(self.phase, model_flux, **kwargs)
        return ax

    def to_phoebe(self):
        pass