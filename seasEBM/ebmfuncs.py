#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:14 2020

@author: andrewpauling
"""


import numpy as np
import scipy.io as io
from scipy.interpolate import CubicSpline


def setupfastM(delx, jmx, D, B, Cl, delt):
    # delx,jmx,D,0.,1.0,delt

    # set up lambda array.
    lam = (1-np.arange(-1, 1.01, delx)**2)/delx**2
    lam = D*lam
    print(lam[0])

    M = np.zeros((jmx, jmx))
    M[0, 0] = -B - lam[1]
    M[0, 1] = lam[1]

    M[jmx-1, jmx-2] = lam[jmx-1]
    M[jmx-1, jmx-1] = -B - lam[jmx-1]

    for j in range(1, jmx-1):
        M[j, j-1] = lam[j]
        M[j, j] = -B - (lam[j+1]+lam[j])
        M[j, j+1] = lam[j+1]

    # add in heat capacities
    M = M/Cl

    return M


def albedo_fdbck_seasonal(T, jmx, x):
    # recalculate albedo.

    alb = np.ones(jmx)*0.3

    # alternative albedo that depends on latitude (zenith angle)
    # alb=0.31+0.08*(3*x.^2-1)/2;

    k = np.argwhere(T <= -5)
    alb[k] = 0.6

    return alb


def daily_insolation(kyear, lat, day, day_type):
    """
    Get daily insolation data.

    Usage:
    Fsw = daily_insolation(kyear,lat,day)

    Optional inputs/outputs:
    [Fsw, ecc, obliquity, long_perh] = daily_insolation(kyear,lat,day,day_type)

    Description:
        Computes daily average insolation as a function of day and latitude at
        any point during the past 5 million years.

   Inputs:
       kyear:    Thousands of years before present (0 to 5000).
       lat:      Latitude in degrees (-90 to 90).
       day:      Indicator of time of year, by default day 1 is Jan 1.
       day_type: Convention for specifying time of year (+/- 1,2) [optional].
       day_type=1 (default): day input is calendar day (1-365.24), where day 1
           is January first.  The calendar is referenced to the vernal equinox
           which always occurs at day 80.
       day_type=2: day input is solar longitude (0-360 degrees). Solar
           longitude is the angle of the Earth's orbit measured from spring
           equinox (21 March). Note that calendar days and solar longitude are
           not linearly related because, by Kepler's Second Law, Earth's
           angular velocity varies according to its distance from the sun.
           If day_type is negative, kyear is taken to be a 3 element array
           containing [eccentricity, obliquity, and longitude of perihelion].

    Output:
        Fsw = Daily average solar radiation in W/m^2.
        Can also output orbital parameters.

    Required file: orbital_parameter_data.mat

    Detailed description of calculation:
        Values for eccentricity, obliquity, and longitude of perihelion for the
        past 5 Myr are taken from Berger and Loutre 1991 (data from   ncdc.noaa.gov). If using calendar days, solar longitude is found using an
        approximate solution to the differential equation representing conservation
        of angular momentum (Kepler's Second Law).  Given the orbital parameters
        and solar longitude, daily average insolation is calculated exactly
        following Berger 1978.

    References:
        Berger A. and Loutre M.F. (1991). Insolation values for the climate of
        the last 10 million years. Quaternary Science Reviews, 10(4), 297-317.
        Berger A. (1978). Long-term variations of daily insolation and
        Quaternary climatic changes. Journal of Atmospheric Science, 35(12),
        2362-2367.

    Authors:
        Ian Eisenman and Peter Huybers, Harvard University, August 2006
        eisenman@fas.harvard.edu
    This file is available online at
    http://deas.harvard.edu/~eisenman/downloads

    For function syntax, enter daily_insolation with no arguments.
    For examples, enter daily_insolation('examples')
    """

    # === Get orbital parameters ===
    if day_type >= 0:
        ecc, epsilon, omega = orbital_parameters(kyear)
        # function is below in this file
    else:
        if len(kyear) != 3:
            print('Error: expect 3-element kyear argument for day_type<0')
            Fsw = np.nan
            ecc = kyear[0]
            epsilon = kyear[1] * np.pi/180
            omega = kyear[2] * np.pi/180

    # For output of orbital parameters
    obliquity = epsilon*180/np.pi
    long_perh = omega*180/np.pi

    # === Calculate insolation ===
    lat = lat*np.pi/180  # latitude
    # lambda (or solar longitude) is the angular distance along Earth's orbit
    # measured from spring equinox (21 March)
    if np.abs(day_type) == 1:  # calendar days
        # estimate lambda from calendar day using an approximation from
        # Berger 1978 section 3
        delta_lambda_m = (day-80)*2*np.pi/365.2422
        beta = (1-ecc**2)**(1/2)
        lambda_m0 = -2*((1/2*ecc+1/8*ecc**3)*(1+beta)*np.sin(-omega) -
                        1/4*ecc**2*(1/2+beta)*np.sin(-2*omega) +
                        1/8*ecc**3*(1/3+beta)*(np.sin(-3*omega)))
        lambda_m = lambda_m0+delta_lambda_m
        lamda = lambda_m+(2*ecc-1/4*ecc**3)*np.sin(lambda_m-omega) + \
            (5/4)*ecc**2*np.sin(2*(lambda_m-omega)) + \
            (13/12)*ecc**3*np.sin(3*(lambda_m-omega))
    elif np.abs(day_type) == 2:  # solar longitude (1-360)
        lamda = day*2*np.pi/360  # lambda=0 for spring equinox
    else:
        raise ValueError('Error: invalid day_type')

    So = 1365  # solar constant (W/m^2)
    delta = np.arcsin(np.sin(epsilon)*np.sin(lamda))  # declination of the sun
    Ho = np.arccos(-np.tan(lat)*np.tan(delta))  # hour angle at sunrise/sunset
    # no sunrise or no sunset: Berger 1978 eqn (8),(9)
    cond1 = np.logical_and(np.abs(lat) >= np.pi/2-np.abs(delta), lat*delta > 0)
    Ho[cond1] = np.pi
    cond2 = np.logical_and(np.abs(lat) >= np.pi/2-np.abs(delta), lat*delta <= 0)
    Ho[cond2] = 0

    # Insolation: Berger 1978 eq (10)
    Fsw = So/np.pi*(1+ecc*np.cos(lamda-omega))**2 / \
        (1-ecc**2)**2*(Ho*np.sin(lat)*np.sin(delta) +
                       np.cos(lat)*np.cos(delta)*np.sin(Ho))

    return Fsw


def orbital_parameters(kyear):
    # === Load orbital parameters (given each kyr for 0-5Mya) ===
    # this .mat file contains the matrix m with data from
    # Berger and Loutre 1991
    data = io.loadmat('orbital_parameter_data.mat')
    m = data['m']
    kyear0 = -m[:, 0]  # kyears before present for data (kyear0>=0);
    ecc0 = m[:, 1]  # eccentricity
    # add 180 degrees to omega (see lambda definition, Berger 1978 Appendix)
    omega0 = m[:, 2]+180  # longitude of perihelion (precession angle)
    # remove discontinuities (360 degree jumps)
    omega0 = np.unwrap(omega0*np.pi/180)*180/np.pi
    epsilon0 = m[:, 3]  # obliquity angle

    # Interpolate to requested dates
    cs1 = CubicSpline(kyear0, ecc0)  # eccs means array of ecc values
    ecc = cs1(kyear)
    cs2 = CubicSpline(kyear0, omega0)
    omega = cs2(kyear) * np.pi/180
    cs3 = CubicSpline(kyear0, epsilon0)
    epsilon = cs3(kyear) * np.pi/180

    return ecc, epsilon, omega






