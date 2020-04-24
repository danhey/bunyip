# -*- coding: utf-8 -*-
from __future__ import division, print_function

import phoebe
import json
import os
import numpy as np
from astropy.stats import BoxLeastSquares
from phoebe.frontend import io
from phoebe.utils import _bytes, parse_json

__all__ = ["Phoetter"]

class Phoetter(phoebe.Bundle):    
    def __init__(self, **kwargs):
        default_bundle_path = os.path.dirname(phoebe.__file__) + '/frontend/default_bundles/default_binary.bundle'
        data = json.load(open(default_bundle_path, 'r'))
        super(Phoetter, self).__init__(params=data)
        
        # Flip constraints and do speedups
        self.__init_bundle()
    
    def __init_bundle(self):
        # Speedups
        phoebe.interactive_constraints_off()
        # self.get_parameter(context='compute', qualifier='irrad_method').set_value("none")
        # self.get_parameter(context='compute', component='primary', qualifier='ntriangles').set_value(300)
        # self.get_parameter(context='compute', component='secondary', qualifier='ntriangles').set_value(300)

        # Add some fitting constraints
        # self.add_constraint('requivratio')
        # self.add_constraint('requivsum')
        self.add_constraint('teffratio')

        # And flip them
        # self.flip_constraint('requivratio', solve_for='requiv@primary')
        # self.flip_constraint('requivsum', solve_for='requiv@secondary')
        self.flip_constraint('teffratio', solve_for='teff@secondary@component')
        self.flip_constraint('esinw', solve_for='ecc')
        self.flip_constraint('ecosw', solve_for='per0')
    
    def model(self, interpolate=True):
        for param, twig in list(self.fit_parameters.items()):
            self[twig] = param
        self.run_compute()
        
        if interpolate:
            return self['fluxes@latest@model'].interp_value(times=time)
        else:
            return self['fluxes@latest@model'].get_value()
        
    def lnlike(self):
        try:
            model_flux = self.model()
            return -0.5 * np.sum((flux - model_flux)**2 / flux_err**2)
        except:
            return -np.inf