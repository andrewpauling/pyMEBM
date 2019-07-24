#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:58:16 2019

@author: andrewpauling
"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import pickle as pkl
import time
from attrdict import AttrDict

# This code solves the energy balance model used in 
# Roe et al. (Nat. Geosci., 2015)
# The model operates in climatology mode
# You can specify:-
# the insolation Q0
# the OLR parameters, A0,B0
# the diffusivity, D
# albedo of ocean and ice
# whether you diffuse moist static energy, or just sensible heat


class MoistEBM():

    def __init__(self, Dmag=0.2598):
        self.timestep = 1./500000
        self.itermax = 10000000

        # Define grid spacing
        jmx = 101
        self.delx = 2.0/jmx

        self.x = self._create_xvec()
        self.phi = np.arcsin(self.x)*180/np.pi

        self.param = AttrDict()

        # climate parameters
        # I think this C = rho * c * h_ml /(pi*1e7).
        # this is consistent with a ~1m layer of dirt
        # Note - heat capacity over LAND for fast convergence
        self.param['Cl'] = 0.2    # units: J /(m2 K)
        self.param['Q0'] = 342    # [W m-2]  solar constant n.b. 4Q0 = 1368
        self.param['A0'] = 207.3  # Size of longwave cooling constant [W/m2]
        self.param['B0'] = 2.09   # [W m-2 degC-1] OLR response NOTE UNITS
        self.param['alf_noice'] = 0.3  # ice free albedo.
        self.param['alf_ice'] = 0.55   # ice covered albedo.

        # Moisture parameters
        self.param['relhum'] = 0.8   # relative humidity
        self.param['eps'] = 0.622    # moisture coonstant
        self.param['psfc'] = 9.8e4   # (Pa)
        self.param['e0'] = 611.2     # vap. press (Pa)
        self.param['a'] = 17.67      # sat vap constants
        self.param['b'] = 243.5      # !!T must be in temperature
        self.param['L'] = 2.45e6     # latent heat of vaporization (J kg-1)
        self.param['cp'] = 1004      # (J kg-1 K-1)

        # magnitude of diffusivity
        self.Dmag = Dmag  # D = 0.2598 W/(m2 K) is the value used by TF10
        self.D = Dmag*np.ones(self.x.size+1)  # diffusivity for MSE

        self.M = self._create_div_matrix()

    def _create_xvec(self):
        x = np.arange(-1.0+self.delx/2, 1.0, self.delx)

        return x

    def _create_div_matrix(self):

        jmx = self.x.size
        # Create matrix to take a divergence of something it acts on
        # set up lambda array.
        lam = (1-np.square(np.arange(-1.0, 1.0+self.delx, self.delx))) / \
            np.square(self.delx)

        lam = np.multiply(self.D, lam)

        M = np.zeros((jmx, jmx))

        M[0, 0] = -lam[1]
        M[0, 1] = lam[1]

        M[jmx-1, jmx-2] = lam[jmx-1]
        M[jmx-1, jmx-1] = -lam[jmx-1]

        for j in range(1, jmx-1):
            M[j, j-1] = lam[j]

            M[j, j] = -(lam[j+1]+lam[j])

            M[j, j+1] = lam[j+1]

        return M  # Divergence matrix

    def _create_t_profile(self):
        tprof = 0.5*(1-1*np.square(self.x))
        return tprof

    def compute(self):

        A = self.param.A0
        B = self.param.B0*np.ones(self.x.size)  # longwave cooling [W/(m2 K)]

        # For dry energy balance model, uncomment these lines:
        # Dmag = 0.44; # magnitude of diffusivity [W/(m2 K)]
        # D=Dmag*np.ones(jmx+1); # diffusivity for sensible (cp*T)
        # relhum = 0;  # switch off humidity

        # set up inital T profile
        T = self._create_t_profile()
        F = np.array([])
        alf0 = self.param.alf_noice*np.ones(self.x.size)
        q0 = self.param['eps']*self.param['relhum']/self.param['psfc'] * \
                self.param['e0']
        dT0 = self.timestep/self.param['Cl']
        Src0 = self.param['Q0']*(1-0.241*(3*self.x*self.x-1))
        
        Src = Src0*(1-alf0)
        
        imbal = (Src - A - B*T).mean()

        # Timestepping loop
        start = time.time()
        while imbal > 0.001:
            
        # for j in np.arange(self.itermax-1):  # use NMAX-1 #for n=1:NMAX
            # Calculate Source  (ASR) for this loop:

            alf = np.where(T <= -10, self.param['alf_ice'], alf0)
#            for idx, item in enumerate(T):
#                if item <= -10:
#                    alf[idx] = self.param.alf_ice

            Src = Src0*(1-alf)

            # spec. hum g/kg, and theta_e
            q = q0*np.exp(self.param['a']*T/(self.param['b']+T))

            theta_e = 1/self.param['cp']*(self.param['cp']*(T+273.15) +
                                          self.param['L']*q)

            # Calculate new T from Source and Sink terms.
            # Diffuse moist static energy (theta_e)
            # hdiv = np.gradient((1-self.x**2)*np.gradient(theta_e, self.x),
            #                   self.x)*self.Dmag
            # dT = self.timestep/self.param.Cl*(Src - A - B*T) + hdiv
            dT = dT0*(Src - A - (B*T) + np.matmul(self.M, theta_e))

            T += dT
            
            imbal = (Src - A - B*T).mean()

#            # Check to see if global mean energy budget has converged:
#            Fglob = np.mean(Src - A - (B*T))
#
#            if np.absolute(Fglob) < 0.001:
#                break

        print('Tmean = '+str(np.mean(T)))
        print('j = '+str(j))
        print('Fglob = '+str(Fglob))

        end = time.time()
        print('time = ' + str(end - start))

        divF = -np.matmul(self.M, theta_e)
        h = theta_e*self.param.cp

        self.T = T
        self.h = h
        self.divF = divF

        self.plot()

    def plot(self):

        matLabData = io.loadmat('ERAtemperature.mat') #loads matlab file as a dictionary with some metadata
        matLabData.keys() #tell me what's in the dictionary
        lat = np.asarray(matLabData['lat'])
        t = np.asarray(matLabData['T'])
        fig = plt.figure(1)
        plt.plot(self.x, self.T)
        plt.plot(np.sin(np.deg2rad(lat)), t)
        plt.xlabel('sin(lat)', fontsize=14)
        plt.ylabel('Temperature', fontsize=14)
        plt.legend(('Model', 'Obs'))
        plt.show()
            
        fig2 = plt.figure(2)
        plt.plot(self.x, self.h)
        plt.xlabel('sin(lat)', fontsize=14)
        plt.ylabel('MSE (J/kg)', fontsize=14)
        plt.show()
    
        #Snk = A+B*T
        plt.plot(self.x, self.divF)
        plt.xlabel('sin(lat)', fontsize=14)
        plt.ylabel('Terms in the energy balance (W/m2)', fontsize=14)
        plt.show()