#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:20:43 2019

@author: andrewpauling
"""

import numpy as np
import time
from pyMEBM.ebm import EBM


class EBMClimo(EBM):
    """
    Class containing an energy balance model in climatology formulation
    """
    def __init__(self, **kwargs):

        super(EBMClimo, self).__init__(**kwargs)

        self.variables['T'] = self._create_t_profile()

        Src0 = self.param['Q0']*(1-0.241*(3*self.x*self.x-1))
        self.variables['Src'] = Src0*(1-self.param['alb_noice'])

    def __repr__(self):
        return super(EBMClimo, self).__repr__()

    def _create_t_profile(self):
        return 0.5*(1-1*self.x*self.x)

    def step(self):
        alf = self.const['alf0'].copy()
        # Calculate Source  (ASR) for this loop:
        #alf = np.where(self.variables['T'] <= -10, self.param['alb_ice'], alf0)
        alf[self.variables['T'] <= -10] = self.param['alb_ice']
        self.variables['Src'] = self.const['Src0']*(1-alf)

        # spec. hum g/kg, and theta_e
        self.variables['q'] = self.const['q0']*np.exp(
                self.const['a']*self.variables['T'] /
                (self.const['b']+self.variables['T']))

        self.variables['theta_e'] = 1/self.const['cp'] *\
            (self.const['cp']*(self.variables['T']+273.15) +
             self.const['L']*self.variables['q'])

        super(EBMClimo, self).step_forward(self.variables['theta_e'],
                                           self.variables['Src'])

    def integrate_converge(self):

        self.const['alf0'] = self.param['alb_noice']*np.ones(self.nx)
        self.const['q0'] = self.const['eps']*self.param['relhum']/self.const['psfc'] * \
            self.const['e0']

        self.const['Src0'] = self.param['Q0']*(1-0.241*(3*self.x*self.x-1))

        imbal = (self.variables['Src'] - self.param['A0'] -
                 self.param['B0']*self.variables['T']).mean()

        print('Integrating to convergence...')
        start = time.time()
        while np.abs(imbal) > 0.01:
            self.step()
            imbal = (self.variables['Src'] - self.param['A0'] -
                     self.param['B0']*self.variables['T']).mean()
        end = time.time()
        print('Done!')
        print('Time = {} seconds'.format(end-start))
