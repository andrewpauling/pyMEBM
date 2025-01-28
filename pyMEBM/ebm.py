#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:58:16 2019

@author: andrewpauling
"""

import numpy as np
import scipy.io as io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pickle as pkl
import time
from numba import njit

# This code solves the energy balance model used in
# Roe et al. (Nat. Geosci., 2015)
# The model operates in climatology mode
# You can specify:-
# the insolation Q0
# the OLR parameters, A0,B0
# the diffusivity, D
# albedo of ocean and ice
# whether you diffuse moist static energy, or just sensible heat


class EBM(object):
    """
    This class sets up an energy balance model with diffusion of either
    sensible heat or moist static energy, as in Roe et al., 2015.

    **Initialization parameters** \n

    An instance of ``EBM`` is initialized with the following
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

    def __init__(
        self,
        diffusion="moist",
        D=None,
        Q0=342.0,
        A0=207.3,
        B0=2.09,
        alb_ice=0.55,
        alb_noice=0.3,
    ):

        self.timestep = 1.0 / 50000
        self.nmax = 60000

        # Define grid spacing
        self.nx = 101
        self.delx = 2.0 / self.nx

        self.x = self._create_xvec()
        self.phi = np.arcsin(self.x) * 180 / np.pi

        # initialize diffusion scheme
        self.diffusion = diffusion

        self.param = {}

        # Initialize diffusion parameter
        if self.diffusion == "moist":
            if D is not None:
                self.param["D"] = D
            else:
                self.param["D"] = 0.2598  # W/(m2 K)

            self.param["relhum"] = 0.8  # relative humidity
        elif self.diffusion == "dry":
            if D is not None:
                self.param["D"] = D
            else:
                self.param["D"] = 0.44  # W/(m2 K)

            self.param["relhum"] = 0.0
        else:
            raise ValueError("Invalid diffusion scheme: '{}'".format(self.diffusion))

        # climate parameters
        # I think this C = rho * c * h_ml /(pi*1e7).
        # this is consistent with a ~1m layer of dirt
        # Note - heat capacity over LAND for fast convergence
        self.param["Q0"] = Q0  # [W m-2]  solar constant n.b. 4Q0 = 1368
        self.param["A0"] = A0  # Size of longwave cooling constant [W/m2]
        self.param["B0"] = B0  # [W m-2 degC-1] OLR response NOTE UNITS
        self.param["alb_noice"] = alb_noice  # ice free albedo.
        self.param["alb_ice"] = alb_ice  # ice covered albedo.
        self.param["Cl"] = 0.2  # units: J /(m2 K)

        self.const = {}
        # Moisture parameters
        self.const["eps"] = 0.622  # moisture coonstant
        self.const["psfc"] = 9.8e4  # (Pa)
        self.const["e0"] = 611.2  # vap. press (Pa)
        self.const["a"] = 17.67  # sat vap constants
        self.const["b"] = 243.5  # !!T must be in temperature
        self.const["L"] = 2.45e6  # latent heat of vaporization (J kg-1)
        self.const["cp"] = 1004  # (J kg-1 K-1)

        # magnitude of diffusivity
        self.D = self.param["D"] * np.ones(self.nx + 1)  # diffusivity for MSE

        self.M = self._create_div_matrix()

        self.variables = {}

        self.variables.theta_e = np.zeros(self.nx)
        self.variables.q = np.zeros(self.nx)

    def __repr__(self):
        summary = ["<pyMEBM.{}>".format(type(self).__name__)]
        summary.append("diffusion: '{}'".format(self.diffusion))
        summary.append("---Parameters---")
        for key, item in self.param.items():
            if key != "relhum" and key != "Cl":
                summary.append("{}: {}".format(key, item))
        return "\n".join(summary)

    def _create_xvec(self):
        return np.arange(-1.0 + self.delx / 2, 1.0, self.delx)

    def _create_div_matrix(self):
        # Create matrix to take a divergence of something it acts on
        # set up lambda array.
        xl = np.arange(-1.0, 1.0 + self.delx, self.delx)
        lam = self.D * (1 - np.square(xl)) / np.square(self.delx)

        M = np.zeros((self.nx, self.nx))

        M[0, 0] = -lam[1]
        M[0, 1] = lam[1]

        M[self.nx - 1, self.nx - 2] = lam[self.nx - 1]
        M[self.nx - 1, self.nx - 1] = -lam[self.nx - 1]

        for j in range(1, self.nx - 1):
            M[j, j - 1] = lam[j]

            M[j, j] = -(lam[j + 1] + lam[j])

            M[j, j + 1] = lam[j + 1]

        return M  # Divergence matrix

    def step_forward(self, theta_e, Src):
        """
        Update temperature given theta_e and Src
        """
        dT0 = self.timestep / self.param["Cl"]
        frc = Src - self.param["A0"] - (self.param["B0"] * self.variables["T"])
        dT = dT0 * (frc + np.matmul(self.M, theta_e))

        self.variables["T"] += dT

    def compute(self):

        A = self.param["A0"]
        B = self.param["B0"] * np.ones(self.x.size)
        # set up inital T profile
        T = self._create_t_profile()

        # Pre-compute some constants for the loop
        alf0 = self.param.alb_noice * np.ones(self.x.size)
        q0 = (
            self.const["eps"]
            * self.param["relhum"]
            / self.const["psfc"]
            * self.const["e0"]
        )
        dT0 = self.timestep / self.param["Cl"]
        Src0 = self.param["Q0"] * (1 - 0.241 * (3 * self.x * self.x - 1))

        Src = Src0 * (1 - alf0)

        imbal = Src - A - B * T
        imbalsum = imbal.mean()

        # Timestepping loop
        start = time.time()
        while imbalsum > 0.1:

            # Calculate Source  (ASR) for this loop:
            alf = np.where(T <= -10, self.param["alb_ice"], alf0)
            Src = Src0 * (1 - alf)

            # spec. hum g/kg, and theta_e
            q = q0 * np.exp(self.const["a"] * T / (self.const["b"] + T))

            theta_e = (
                1
                / self.const["cp"]
                * (self.const["cp"] * (T + 273.15) + self.const["L"] * q)
            )

            # Calculate new T from Source and Sink terms.
            # Diffuse moist static energy (theta_e)
            dT = dT0 * (Src - A - (B * T) + np.matmul(self.M, theta_e))

            T += dT

            imbal = Src - A - B * T
            imbalsum = imbal.sum()

        print("Tmean = " + str(np.mean(T)))
        print("Fglob = " + str(imbal))

        end = time.time()
        print("time = " + str(end - start))

        divF = -np.matmul(self.M, theta_e)
        h = theta_e * self.const["cp"]

        self.T = T
        self.h = h
        self.divF = divF
        self.imbal = imbal

        # self.plot()

    def plot(self):

        # load matlab file as a dictionary with some metadata
        matLabData = io.loadmat("ERAtemperature.mat")
        matLabData.keys()  # tell me what's in the dictionary
        lat = np.asarray(matLabData["lat"])
        t = np.asarray(matLabData["T"])
        fig = plt.figure(1)
        plt.plot(self.x, self.variables["T"])
        plt.plot(np.sin(np.deg2rad(lat)), t)
        plt.xlabel("sin(lat)", fontsize=14)
        plt.ylabel("Temperature", fontsize=14)
        plt.legend(("Model", "Obs"))
        plt.show()

        fig2 = plt.figure(2)
        plt.plot(self.x, self.variables["theta_e"])
        plt.xlabel("sin(lat)", fontsize=14)
        plt.ylabel("MSE (J/kg)", fontsize=14)
        plt.show()

        plt.plot(self.x, self.divF)
        plt.xlabel("sin(lat)", fontsize=14)
        plt.ylabel("Terms in the energy balance (W/m2)", fontsize=14)
        plt.show()
