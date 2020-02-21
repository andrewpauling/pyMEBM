#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:24:46 2020

@author: andrewpauling
"""


import numpy as np
from attrdict import AttrDict


class EBM(object):
    
    def __init__(self,
                 diffusion='moist'):

        self.diffusion = diffusion

        self.timestep = 1./500000

        # Define grid spacing
        self.nx = 101
        self.delx = 2.0/self.nx

        self.x = self._create_xvec()
        self.phi = np.arcsin(self.x)*180/np.pi

        self.const = AttrDict()
        self.param = AttrDict()

        # Set constants
        self.const['psfc'] = 1.013e5
        self.const['g'] = 9.81
        self.const['Re'] = 6.37e6
        self.const['cp'] = 1004
        self.const['Lv'] = 2.45e6

        # Thermodynamics and moisture parameters
        self.const['eps'] = 0.622
        self.const['e0'] = 611.2
        self.const['a'] = 17.67
        self.const['b'] = 243.5

        self.param['A0'] = 201
        self.param['B0'] = 2.09

        if self.diffusion == 'moist':
            self.param['relhum'] = 0.8
            D_HF10 = 1.06e6
        elif self.diffusion == 'dry':
            self.param['relhum'] = 0.0
            D_HF10 = 1.70e6
        else:
            raise ValueError(
                "Invalid diffusion scheme '{}'. ".format(self.diffusion) +
                "Please use either 'dry' or 'moist'")

        self.param['D'] = D_HF10*self.const['psfc']*self.const['cp'] / \
            (self.const['g']*self.const['Re']**2)

    def __repr__(self):
        summary = ['<pyMEBM.{}>'.format(type(self).__name__)]
        summary.append("diffusion: '{}'".format(self.diffusion))
        summary.append("---Parameters---")
        for key, item in self.param.items():
            if key != 'relhum' and key != 'Cl':
                summary.append('{}: {}'.format(key, item))
        return '\n'.join(summary)

    def _create_xvec(self):
        return np.arange(-1.0+self.delx/2, 1.0, self.delx)

    def _create_div_matrix(self):
        # Create matrix to take a divergence of something it acts on
        # set up lambda array.
        xl = np.arange(-1.0, 1.0+self.delx, self.delx)
        lam = self.D*(1-np.square(xl))/np.square(self.delx)

        M = np.zeros((self.nx, self.nx))

        M[0, 0] = -lam[1]
        M[0, 1] = lam[1]

        M[self.nx-1, self.nx-2] = lam[self.nx-1]
        M[self.nx-1, self.nx-1] = -lam[self.nx-1]

        for j in range(1, self.nx-1):
            M[j, j-1] = lam[j]

            M[j, j] = -(lam[j+1]+lam[j])

            M[j, j+1] = lam[j+1]

        return M  # Divergence matrix
        
        