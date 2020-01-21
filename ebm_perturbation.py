#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:49:34 2019

@author: andrewpauling
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict
import scipy.io as io
from scipy.interpolate import griddata

from pyMEBM.ebm import EBM


class EBMPert(EBM):
    """
    This class sets up an energy balance model with diffusion of either
    sensible heat or moist static energy, as in Roe et al., 2015.

    **Initialization parameters** \n

    An instance of ``EBMPert`` is initialized with the following
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

    :param str forcing:         flag to specify forcing to be used. Options
                                for now are 'flat' (default) or 'CMIP5'

    :param str feedbacks:       flag to specify feedbacks to be used. Options
                                for now are 'flat' (default) or 'CMIP5'
                                
    :param bool ocn_heat_uptake: flag to include ocean heat uptake as part of
                                forcing. False by default
                                
    """
    def __init__(self,
                 ocn_heat_uptake=False,
                 forcing='flat',
                 feedbacks='flat',                 
                 **kwargs):
        
        self.ocn_heat_uptake = ocn_heat_uptake
        self.feedbacks = feedbacks
        self.forcing = forcing

        super(EBMPert, self).__init__(**kwargs)
        
        self.variables['T'] = 5*np.ones(self.nx)
        
        # Initialize forcing and feedback parameters
        self._load_forcing_feedbacks()
        self._load_ctrl_data()

    def __repr__(self):
        summary = [super(EBMPert, self).__repr__()]
        summary.append('---Configuration___')
        summary.append('forcing: {}'.format(self.forcing))
        summary.append('feedbacks: {}'.format(self.feedbacks))
        summary.append('ocn_heat_uptake: {}'.format(self.ocn_heat_uptake))
        return '\n'.join(summary)

    def _load_ctrl_data(self):
        # loads matlab file as a dictionary with some metadata
        matLabData = io.loadmat('ERAtemperature.mat')

        lat = np.asarray(matLabData['lat'])
        T_ctrl = np.asarray(matLabData['T'])
        # average N & S hemispheres for symmetry
        T_ctrl = 0.5*(T_ctrl+np.flipud(T_ctrl))
        T_ctrl = griddata(np.sin(np.deg2rad(lat)), T_ctrl, self.x,
                          method='linear')
        T_ctrl = np.squeeze(T_ctrl, axis=1)
        # here T is in oC. q is g kg-1
        q_ctrl = self.const['eps']*self.param['relhum']/self.const['psfc'] *\
            self.const['e0']*np.exp(
                    self.const['a']*(T_ctrl)/(self.const['b']+(T_ctrl)))
        theta_e_ctrl = 1/self.const['cp'] * \
            (self.const['cp']*((T_ctrl)+273.15) + self.const['L']*q_ctrl)

        self.variables['T_ctrl'] = T_ctrl
        self.variables['theta_e_ctrl'] = theta_e_ctrl

    def _load_forcing_feedbacks(self):
        # CMIP5 ensemble-mean feedback and forcing values from 4xCO2
        # simulations (taken at year 100)
        # feedback, forcing and heat uptake for 11 models
        matLabData = io.loadmat('CMIP5_Rf_G_lambda.mat')
        CMIP5_lat = np.asarray(matLabData['CMIP5_lat'])  # latitude
        CMIP5_T = np.asarray(matLabData['CMIP5_T'])  # Temperature change
        CMIP5_lambda = np.asarray(matLabData['CMIP5_lambda'])  # feedbacks
        CMIP5_Rf = np.asarray(matLabData['CMIP5_Rf'])  # Radiative forcing
        CMIP5_G = np.asarray(matLabData['CMIP5_G'])  # Ocean heat uptake
        CMIP5_names = np.asarray(matLabData['CMIP5_names'])

        if self.forcing == 'flat':
            # uniform forcing in [W/m2] for a quadrupling of CO2, 7.8 value
            # taken as global average of CMIP5
            R_frc = 7.8  # W/m^2
        else:
            if self.ocn_heat_uptake:
                frc = CMIP5_Rf + CMIP5_G
            else:
                frc = CMIP5_Rf

            R_frc = griddata(CMIP5_lat, np.mean(frc, axis=1), self.phi,
                             method='linear')

        if self.feedbacks == 'flat':
            # uniform feedback [W/m2/K], 1.4 value taken as average of CMIP5
            B = 1.4*np.ones(self.nx)
        else:
            B = -griddata(CMIP5_lat, np.mean(CMIP5_lambda, axis=1), self.phi,
                          method='linear')  # taking average over CMIP5 models

        self.R_frc = R_frc
        self.lamparam = B

    def step(self):

        # Pre-compute some constants for the loop
        q0 = self.const['eps']*self.param['relhum']/self.const['psfc'] * \
            self.const['e0']

        # spec. hum g/kg, and theta_e
        self.variables['q'] = q0 * \
            np.exp(self.const['a']*(self.variables['T']+self.variables['T_ctrl']) /
                   (self.const['b']+self.variables['T']+self.variables['T_ctrl']))

        self.variables['theta_e'] = 1/self.const['cp'] *\
            (self.const['cp']*(self.variables['T']+self.variables['T_ctrl']+273.15) +
             self.const['L']*self.variables['q'])

        theta_e_pert = self.variables['theta_e'] - self.variables['theta_e_ctrl']

        Src = self.param['A0'] + \
            self.param['B0']*(self.variables['T'])
        Src += self.R_frc - self.lamparam*self.variables['T']

        super(EBMPert, self).step_forward(theta_e_pert, Src)

    def integrate_converge(self):
        imbal = (self.R_frc - self.lamparam*self.variables['T']).mean()

        print('Integrating to convergence...')
        start = time.time()
        while np.abs(imbal) > 0.001:
            self.step()
            imbal = (self.R_frc - self.lamparam*self.variables['T']).mean()
        end = time.time()
        print('Done!')
        print('Time = {} seconds'.format(end-start))
