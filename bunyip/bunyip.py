# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .knn import KNN
from .geometry import solve_geometry, from_geometry

import numpy as np
import ellc
from scipy import optimize
import lightkurve as lk
import matplotlib.pyplot as plt
import tqdm

__all__ = ["Bunyip"]

class Bunyip:
    def __init__(self, phase, flux, flux_err=None):

        # Assign initial values
        self.phase = phase
        self.flux = flux / np.median(flux)
        if flux_err is None:
            flux_err = np.zeros_like(flux)
        self.flux_err = flux_err

        # Initialise fit parameters
        self.parameters = {
            'q':    1.,
            'rsum': 0.5,
            'rratio': 0.5,
            'fc': 0,
            'fs': 0,
            'sbratio': 1.,
            'incl': 89.,
            't0': 0.,
            'mean': 0.,#np.mean(self.flux) - 1.,
            'log_f': np.std(self.flux),
        }

        # Parameters specific to ellc
        self.ellc_parameters = {
            'shape_1': shape_1,
            'shape_2': shape_2
        }
        
        # Some wrappers for the optimizers. don't @ me
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
        """
        if period is None:
            raise ValueError('I really should calculate the period here.')
        if flux_err is None:
            flux_err = np.zeros_like(time)

        # Fold and sort the light curve
        phase = (t0 % period) / period    
        fold_time = (((time - phase * period) / period) % 1)
        fold_time[fold_time > 0.5] -= 1
        sorted_args = np.argsort(fold_time)

        phase, phase_flux, phase_flux_err = fold_time[sorted_args], flux[sorted_args], flux_err[sorted_args]
        if flux_err is None:
            flux_err = np.zeros_like(time)
        
        if bins is not None:
            lc = lk.LightCurve(fold_time[sorted_args], flux[sorted_args])
            binned = lc.bin(bins=bins)
            phase, phase_flux = binned.time, binned.flux
            phase_flux_err = np.zeros_like(phase)

        # Bunyip needs args
        return Bunyip(phase, phase_flux, phase_flux_err)

    def lc_model(self):
        rsum = self.parameters['rsum']
        rratio = self.parameters['rratio']
        r1, r2 = rsum/(1.+rratio), rsum*rratio/(1.+rratio)

        lc = ellc.lc(self.phase,
                     t_zero=self.parameters['t0'],
                     q=self.parameters['q'],
                     radius_1=r1, 
                     radius_2=r2,
                     incl=self.parameters['incl'],
                     sbratio=self.parameters['sbratio'], 
                     f_c=self.parameters['fc'], 
                     f_s=self.parameters['fs'],
                     shape_1=self.ellc_parameters['shape_1'], 
                     shape_2=self.ellc_parameters['shape_1'],
                    )
                    
        return lc + self.parameters['mean']

   
    def update_from_geometry(self, **kwargs):
        dphi = solve_geometry(self.phase, self.flux, **kwargs)
        ecc, per0 = from_geometry(np.abs(dphi))
        self.parameters.update({
            'fc': np.sqrt(ecc) * np.cos(per0),
            'fs': np.sqrt(ecc) * np.sin(per0),
        })

    def update_from_knn(self):
        model = KNN()
        lc = lk.LightCurve(self.phase, self.flux)
        binned = lc.bin(bins=101).normalize()
        prediction = model.predict(binned.flux)[0]
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

    def fit_spline(xs=None):

        pass

    def update_parameters(self, prediction):
        """Update parameters from a prediction given by either the neural
        network, or the KNN classifier
        
        Parameters
        ----------
        prediction : list
            List of prediction values, should be in the form 
            q, r1, r2, tratio, incl, ecc, per0
        """
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
        """log prior of the ellc model
        
        Returns
        -------
        float
            The prior probability
        """
        rsum = self.parameters['rsum']
        rratio = self.parameters['rratio']
        r1, r2 = rsum/(1.+rratio), rsum*rratio/(1.+rratio)
        
        ecc = self.parameters['fc']**2 + self.parameters['fs']**2
        per0 = np.arctan2(self.parameters['fs'],self.parameters['fc'])
        if ((0 < self.parameters['q']) & (self.parameters['incl'] <= 90) & (0 < self.parameters['sbratio']) & 
            ((-0.5) <= self.parameters['t0'] <= (0.5)) & (0 < r1 < 1) & (0 < r2 < 1) & (0 <= ecc < 1.)):
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
        """Global optimization with differential evolution. Costly, but effective!
        
        Parameters
        ----------
        bounds : list
            List of bounds for each parameters (i.e. [(0,1), (2,5)])
        
        Returns
        -------
        dict
            Results of the optimization
        
        Raises
        ------
        ValueError
            As many bounds as parameters must be passed.
        """
        vars = list(self.parameters.keys())

        if not len(vars) == len(bounds):
            raise ValueError("You must have as many bounds as parameters for \
                            differential evolution")
        
        soln = optimize.differential_evolution(self.nll_wrapper, bounds, args=(vars))
        for val, key in zip(soln.x, vars):
            self.parameters.update({key:val})
        return soln
        
    def plot_model(self, ax=None, **kwargs):
        """Plot the current model in the parameters dict
        
        Parameters
        ----------
        ax : matplotlib axis, optional
            axis object on which to plot, by default None
        
        Returns
        -------
        matplotlib axis
            axis
        """
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
        """Run emcee initialised around the current parameters
        
        Parameters
        ----------
        burnin : int, optional
            Number of burn-in values, by default 1000
        draws : int, optional
            Number of draws, by default 2000
        nwalkers : int, optional
            Number of walkers in the chain, by default 32
        
        Returns
        -------
        [type]
            [description]
        """
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

    def get_model_from_trace(self, trace, n=100):
        vars = list(self.parameters.keys())
        samples = trace[np.random.choice(len(trace), size=n)]
        lcs = []
        for i in samples:
            [self.parameters.update({key:val}) for key, val in zip(vars, i)]
            model_flux = self.lc_model()
            lcs.append(model_flux)
        return lcs

    def optimize_best(self, **kwargs):
        optimization_path = [
            ['t0'],
            ['mean', 'log_f'],
            ['t0', 'mean', 'log_f', 'fc', 'fs'],
            ['t0', 'mean', 'log_f', 'fc', 'fs', 'rsum', 'rratio', 'incl', 'mean'],
            None # All parameters
        ]

        for path in tqdm.tqdm(optimization_path):
            soln = self.optimize(vars=path, method='Nelder-Mead', **kwargs)
        return soln



    # def to_phoebe(self, time, flux):
    #     from .phoetter import Phoetter
    #     p = Phoetter()
    #     p.add_dataset('lc', times=time, fluxes=flux)#np.zeros_like(self.flux), fluxes=self.flux)
    #     p.flip_constraint('compute_phases', 'compute_times')
    #     p.set_value('compute_phases', self.phase)

    #     p.set_value('period@orbit', self.period)
    #     p['q'] = self.parameters['q']

    #     rsum = self.parameters['rsum']
    #     rratio = self.parameters['rratio']
    #     r1, r2 = rsum/(1.+rratio), rsum*rratio/(1.+rratio)
    #     p['requiv@primary'] = p['value@sma@binary']*r1
    #     p['requiv@secondary'] = p['value@sma@binary']*r2
    #     p['incl@orbit'] = self.parameters['incl']
    #     p['t0_supconj'] = self.parameters['t0'] * self.period
    #     p['teffratio'] = (self.parameters['sbratio'] / (self.parameters['rratio'] **2)) **(1/4)

    #     ecc = self.parameters['fc']**2 + self.parameters['fs']**2
    #     per0 = np.arctan2(self.parameters['fs'],self.parameters['fc'])

    #     p['ecosw'] = ecc*np.cos(per0)
    #     p['esinw'] = ecc*np.sin(per0)

    #     p['pblum_mode'] = 'dataset-scaled'
    #     p.set_value_all('atm', 'blackbody')
    #     p.set_value_all('ld_mode', 'manual')
    #     p.set_value_all('ld_mode_bol', 'manual')

    #     print(p.run_checks())
    #     return p