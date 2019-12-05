# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

# __all__ = ["QuadraticLimbDarkening", "LinearLimbDarkening"]

class LimbDarkening:
    def __init__(self):
        pass

class QuadraticLimbDarkening(LimbDarkening):
    pass

    def q_to_u(self, q1, q2):
        return 2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)

class LinearLimbDarkening(LimbDarkening):
    pass