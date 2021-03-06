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
    This class sets up an energy balance model with diffusion of either
    sensible heat or moist static energy, as in Roe et al., 2015.

    **Initialization parameters** \n

    An instance of ``EBMClimo`` is initialized with the following
    arguments *(for detailed information see Object attributes below)*:

    :param str diffusion:       string that sets the type of diffusion to be
                                used. Valid values: 'dry' or 'moist'

                                - default value: ``'moist'``

    :param float D:             value of the diffusion coefficient \n
                                unit: :math:`\\textrm{W m}^{-2}`

                                - default value: ``0.2598 (moist), 0.44 (dry)``

    :param float Q0:            value of the solar insolation    \n
                                unit: :math:`\\textrm{W m}^{-2}`

                                -default value: ``342.0``

    :param float A0:            value of longwave parameter A in A + BT  \n
                                unit: :math:`\\textrm{W m}^{-2}`

                                -default value: ``207.3``

    :param float B0:            value of longwave parameter B in A + BT  \n
                                unit: :math:`\\textrm{W m}^{-2}\\textrm{K}^{-1}`

                                -default value: ``2.09``

    :param float alb_ice:       value of ice albedo  \n
                                unit: ``unitless``

                                -default value: ``0.55``

    :param float alb_noice:     value of non-ice albedo  \n
                                unit: ``unitless``

                                -default value: ``0.3``
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
        """
        Integrate the climatological EBM to convergence.

        Returns
        -------
        None.

        """

        self.const['alf0'] = self.param['alb_noice']*np.ones(self.nx)
        self.const['q0'] = self.const['eps']*self.param['relhum'] / \
            self.const['psfc']*self.const['e0']

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
