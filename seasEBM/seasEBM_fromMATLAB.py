#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:45:25 2020

@author: andrewpauling
"""


import numpy as np
import scipy.io as io
from ebmfuncs import setupfastM, albedo_fdbck_seasonal, daily_insolation
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = 12, 8
mpl.rcParams['font.size'] = 18

"""
Code for Climatological Energy Balance Model
GHR 13Oct2014. Based on old ebm written by GHR and CMB
New code GHR Feb18
Will have:
- ability to have seasonal cycle - yes
- ability to have orbital forcing - yes
- two versions of albedo feedback - yes
- latitude-dependent heat capacity - yes
- coupling to 2-layer model, can be specified as function of latitude - yes
- A, B, D can be specified as function of latitude - yes
- ability to have noise - yes
"""

# set parameters and display their values
print('*************************************************')
print('*** Climatological Moist Energy Balance Model ***')
print('*************************************************')
print('This calculates a climatology, for simple albedo feedback')
print('!!This is a terribly written code!!')
print('It does have a convergence criterion built in,')
print('but check that Sum = 0 to your satisfaction in the top panel')

# Physical constants
psfc = 1.013e5  # [Pa] surface pressure
g = 9.81        # [m s^-2] er....
Re = 6.37e6     # [m] Earth's radius
cp = 1004       # [K kg^-1 K^-1] specific heat constant pressure
Lv = 2.45e6     # [J kg-1] latent heat of vaporization (J kg-1)

# Thermodynamics and moisture parameters
relhum = 0.8   # relative humidity
eps = 0.622    # moisture coonstant (Rd/Rv I think)
e0 = 611.2     # vap. press (Pa)
a = 17.67
b = 243.5      # sat vap constants !!T must be in temperature

# Time array
ts = 0     # [yr] Initial time
tf = 50.0  # [yr] Final time

# time step in fraction of year
# make denominator an integer multiple of 360
# if code blows up, try shortening the time step (this is explicit scheme)
delt = 1./(360*6)
print(f'delt = {delt}')  # [yr^-1] trial and error 140 time steps a day.

# number of time steps
nts = int((tf-ts)/delt + 1)

# time indicators
t = ts             # [yr] time years (could be past climate for Milankovitch)
yr = 0             # [yr] model years, updated immediately time loops starts
mnth = 1           # [month] model months
day = 1            # day of year (360 days in a year)
newday_flg = 1     # is it new day? set to yes (=1) initially
newmonth_flg = 1   # is it a new month? set to yes (=1) initially

# other flags
season_flg = 1          # seasonal cycle? (1=yes, 0=no), day 1 = vernal equinox
alb_flg = 1             # albedo temperature feedback? (1=yes, 0 = no)
symmetric_radn_flg = 0  # symmetrize insolation about equator? (1=yes, 0=no)
two_layer_flg = 1       # two layer ocean model? (1=yes, 0=no)
noise_flg = 0           # noise flag (can be specified in many ways)

# set up x array (latitude).
jmx = 101
delx = 2.0/jmx
x = np.arange(-1.0+delx/2, 1.0-delx/2+0.001, delx)  # sine latitude
phi = np.arcsin(x)*180/np.pi                    # latitude

# climate parameters
# A=203.3*ones(size(x)); A=A(:); % [W/m2] Size of longwave cooling constant
# disp('A is tweaked for new insolation');
A = 201.0*np.ones(x.size)  # [W/m2] Size of longwave cooling constant
print(f'A = {np.mean(A)} W m-2')
B = 2.09*np.ones(x.size)   # [W/(m2 K)] longwave radiative damping
print(f'B = {np.mean(B)} W m-2 K-1')


# diffusivity; units conversion is always tricky
D_HF10 = 1.06e6  # [m^2 s^-1] Hwang and Frierson (GRL, 2010)

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
For DRY EBM, uncomment these values
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# D_HF10 = 1.70e6;                % diffusivity for sensible heat.
# rel_hum = 0;                    % switch off latent-heat dependence.

# convert diffusivity units for this numerical code
# conversion to regular units here. For HF10, D~0.28
# For dry EBM see Armouretal19, but should be ~0.44
Dmag = D_HF10*psfc*cp/(g*Re**2)  # [W m^-2 K^-1]
print(f'D = {Dmag}  W/(m2 K)')  # D = 0.2598 W/(m2 K) is the value used by TF10

D = Dmag*np.ones(jmx+1)       # diffusivity array (at each cell edge)
D_mid = 0.5*(D[:-1] + D[1:])  # diffusivity at cell midpoing.

# heat capacity, mixed layer depth
# two-layer coupling parameter (2 layer = 1, 1 layer = 0)
if two_layer_flg:
    gamma = 0.67*np.ones(x.size)  # [W m^-2 K-1]
    # coupling b/t surface and deep ocean, based on CMIP 5 fit from
    # Armour (NCC 2017)
    print('!!!!!!!! Using two layer ocean WATCH FOR DRIFT!!!!!!')
else:
    gamma = np.zeros(x.size)  # no coupling b/t surface and deep ocean

print(f'two-layer coupling gamma = {np.mean(gamma)} W m-2 K-1')

mix_depth = 35   # [m] mixed-layer depth
# depth of mixed layer (assuming water), can be specified with latitude.
h_ml = mix_depth*np.ones(x.size)  # [m]
rho_w = 1e3  # [kg m^-3]
cw = 4200    # [J kg^-1 K^-1] heat capacity of water

# note: units of heat capacity allow units of years for time
C_L1 = rho_w*cw*h_ml/(np.pi*1e7)  # heat capacity of layer 1 (upper layer)
print(f'h_mix_lyr = {mix_depth} m; C_L1 = {np.mean(C_L1)} J /(m2 K s yr^-1)')

# deep layer for two-layer ocean model
# deep-layer depth based roughly on CMIP5 fit from Armour(NCC, 2017)
h_d = 800  # [m]
h_ml = mix_depth*np.ones(x.size)  # [m] depth of deep-ocean layer
C_L2 = rho_w*cw*h_d/(np.pi*1e7)  # heat capacity of layer 2 (upper layer)
print(f'h_deep = {h_d} m; C_L2 = {np.mean(C_L2)} J /(m2 K s yr^-1)')

# daily or annual mean insolation from Huybers and Eisenman
# calling it just once here, outside the main timeloop, is much faster...
kbp = 0  # orbital year in units of kbp
# insolation for every day; store as an array for computational speed
# Note day1 = vernal equinox
Sday = np.zeros((360, jmx))
for i in range(360):
    Sday[i, :] = daily_insolation(kbp, phi, i, 2)  # daily insolation


# make radiation symmetric about equator? (i.e., lose eccentricity effect)?
if symmetric_radn_flg:
    # phase shift season by 6 months
    tmp = np.concatenate((Sday[180:, :], Sday[:180, :]), axis=0)
    tmp = np.fliplr(tmp)  # reflect insol pattern about the equator
    # average both patterns to symmetrize radiation about equator
    Sday = 0.5*(Sday+tmp)


# if no seasonal cycle replace Sday with annual-mean values
# n.b. b/c eccentricity, not symmetric. Can symmetrize if desired.
if season_flg == 0:  # no annual cycle
    tmp = np.mean(Sday, axis=0)
    for i in range(360):
        Sday[i, :] = tmp

# Legendre polynomial realization of mean annual insol.
# Q = 340.0; disp(['Q = ' num2str(Q) ' W/m2']); %0.25*solar constant [W/m2], small compated to some values
# S = Q*(1-0.241*(3*x.^2-1)); S=S(:);

# set up inital T profile; simple guess to facilitate convergence.
if two_layer_flg == 0:  # one layer only
    T_init = 25*(1-2*x**2)
    T = T_init
else:                                  # two layer - load initial profile
    data = io.loadmat('T_init.mat')  # two layer. use previous equilibrium
    T_init = np.squeeze(data['T_init'])
    x_init = np.squeeze(data['x_init'])
    T = np.interp(x, x_init, T_init)

Td = T  # give deep ocean temperature the initial temperature

# Use setupfast to create a matrix that calculates D*d/dx[ (1-x^2)d/dx] of
# whatever it operates on.
Mdiv = setupfastM(delx, jmx, D, 0., 1.0, delt)

# albedo
if alb_flg:
    if season_flg:
        alb = albedo_fdbck_seasonal(T, jmx, x)
        print('Seasonal albedo feedback included')
    else:
        alb = albedo_fdbck_seasonal(T, jmx, x)  # seasonal value T(ice)<-5C
else:
    alb = 0.31+0.08*(3*x**2-1)/2
    print('No albedo feedback')

# Global mean temperature
Tglob = np.mean(T)

Tout = np.zeros((int(tf), 12, jmx))
Tdout = np.zeros((int(tf), 12, jmx))

# Timestepping loop
for n in range(nts-1):   # the -1 stops the loop before getting to yr_end +1
    Tglob_prev = Tglob

    # time indexing, output every month
    t = ts + n*delt  # [yrs] time in years

    # need to update days, months, and years
    # is it a new year?
    tmp = np.floor(t) + 1

    if tmp != float(yr) or n == 0:
        yr += 1
        print(f'happy new year! yr = {yr}')
        # add interannual noise
        if noise_flg:
            # different random noise at each latitude
            noise = 15*np.rand.randn(jmx)  # [W m-2]
        else:
            noise = np.zeros(jmx)
    # is it a new month?
    # takes the fraction of the year and calulates the month from 1 to 12
    tmp = np.floor((t-np.floor(t))*(12))+1
    if tmp != mnth:
        if mnth < 12:
            mnth += 1
        else:
            mnth = 1
        newmonth_flg = 1
        # print(f'its a new month! mnth = {mnth}')

    # is it a new day?
    # takes the fracation of the year and calulates the day from 1 to 360
    tmp = np.floor((t-np.floor(t))*(360))+1
    if tmp != day:
        if day < 360:
            day += 1
        else:
            day = 1

        newday_flg = 1
        # print(f'its a new day! day = {day}')

    # albedo
    if alb_flg:
        if season_flg:
            alb = albedo_fdbck_seasonal(T, jmx, x)
        else:
            alb = albedo_fdbck_seasonal(T, jmx, x)  # seasonal value T(ice)<-5C

    else:
        alb = 0.31+0.08*(3*x**2-1)/2
        print('No albedo feedback')

# -------------------------------------------------------------------
    # if new day, calculate insolation if we're doing the seasonal cycle
    if newday_flg:
        S = Sday[day-1, :]
        newday_flg = 0  # set new day flag to zero

    # Calculate src for this loop.
    src = ((1-alb)*S-A)/C_L1

    # spec. hum, and theta_e
    q = eps*relhum/psfc*e0*np.exp(a*T/(b+T))  # here T is in oC. q is g kg-1
    theta_e = 1/cp*(cp*(T+273.15) + Lv*q)     # note units of K are needed!!!

    # Calculate new T.

    #   Uncomment for diffusion of dry static energy
    #   dT = delt/Cl*((1-alb).*S - (A+B.*T) + Mdiv*T + gamma*(Td-T)); if n == 1,disp(['Diffusing T']),end;

    # Diffuse moist static energy (theta_e)
    dT = delt/C_L1*((1-alb)*S - (A+B*T) + np.matmul(Mdiv, theta_e) +
                    gamma*(Td-T)+noise)

    dTd = delt/C_L2*gamma*(T-Td)  # deep temperature update

    # update temperature
    T += dT
    Td += dTd

    """
    --------------------------------
    if new month, output data
    --------------------------------
    """
    if newmonth_flg == 1:
        Tout[yr-1, mnth-1, :] = T
        Tdout[yr-1, mnth-1, :] = Td   # can be junked if using ony 1layer
        newmonth_flg = 0         # set new day flag to zero


"""
Hydrocycle and graph output
What is below just graphs the final state of the model
Monthly output from the model is stored in
Tout (surface layer), and Tdout (deep-ocean layer)
Everything else (probably) can be diagnosed or added to the output
"""
# climatological fields
divFtot = -np.matmul(Mdiv, theta_e)  # divergence of total flux
src = (1-alb)*S                     # absorbed solar radn
h = theta_e*cp                      # surface mse (moist enthalpy)

# total flux, note the factors to get units of W
Ftot = -2*np.pi*Re**2/cp*D_mid*(1-x**2)*np.gradient(h, x)

"""
Diagnose terms hydrological cycle
exp weighting function for Hadley Cell (Siler et al., 2018)
"""
sigx = 0.13                   # width of weighting function
wt = np.exp(-(x/sigx)**2)     # weighting function for Hadley Cell

# heq = h(x==0);                     % moist static energy at the equator
hmax = np.max(h)                     # find maximum h value
imax = np.argwhere(h == hmax)        # index value of max h;
# if 2 locations found, take the first and issue a warning.
if len(imax) > 1:
    imax = imax[0]
    print('****Caution - more than one max-h location****')

xmax = x[imax]                     # x location of max h

# gross moist stability. 6% of surface value (Siler et al., 2018)
dh_gms = 0.06*hmax

F_hc = wt*Ftot                      # Hadley cell flux
V = F_hc/(hmax+dh_gms-h)           # Diagnosed mass transport in Hadley Cell
V[imax] = 0
# eddy latent heat fluxes including weighting function.
F_LH_eddy = (1-wt)*-2*np.pi*Re**2/cp*D_mid*(1-x**2)*np.gradient(Lv*q, x)
# total laten heat flux, including equatorward component due to Hadlecy Cell
F_LH = -Lv*V*q + F_LH_eddy

divF_LH = 1/(2*np.pi*Re**2)*np.gradient(F_LH, x)
E_m_P = divF_LH/(Lv*rho_w)*np.pi*1e7                 # E-P in m/yr

F_DSE = Ftot-F_LH          # dry static energy

"""
code from Nick Siler to partition E-P;
Based on Siler et al., 2018, and annual-mean parameterization
Should talk to Nick about using these for the seasonal cycle
"""
alpha = Lv/461.5/(T+273.15)**2                       # Nick's alpha parameter
beta = cp/Lv/alpha/q                                 # beta parameter

# fix for seasonal cycle
# [W m-2] made-up idealized pattern of R-G
RG = 180*(1*(1-(x-xmax)**2)-.4*np.exp(-((x-xmax)/.15)**2))

Ch = 1.5e-3                                # drag coefficient
LWfb = 0                                   # LW feedback at surface, in W/m2/K

# fix for seasonal cycle
u = 4+np.abs(np.sin(np.pi*(x-xmax)/1.5))*4  # wind speed

rho_air = 1.2                       # air density
Evap = (RG*alpha+rho_air*cp*(1-relhum)*Ch*u)/(alpha+cp/Lv/q)  # [W m-2] evap
Prec = Evap - divF_LH                                         # [W m-2] precip
Prec = np.squeeze(Prec/(Lv*rho_w)*np.pi*1e7)  # [m yr-1]
Evap = np.squeeze(Evap/(Lv*rho_w)*np.pi*1e7)  # [m yr-1]


# graphic check on the energy balance
fig = plt.figure(1, figsize=(24, 16))
# plot as equal area (sine of latitude)
# xt = [-60 -45 -30 -15 0 15 30 45 60];
# xtl = ['-60'; '-45'; '-30'; '-15'; '  0'; ' 15'; ' 30'; ' 45'; ' 60'];
# xt = np.sind(xt)

# ---------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(phi, T+273.15, linewidth=2)
ax1.plot(phi, h/cp, linewidth=2)
ax1.legend(('T', 'h/c_p'))
ax1.set_ylabel('K')
ax1.set_title('T and h')
ax1.grid()

# ---------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(phi, Ftot, linewidth=2)
ax2.plot(phi, F_LH, linewidth=2)
ax2.plot(phi, F_DSE, linewidth=2)
ax2.grid()
ax2.set_ylabel('PW')
ax2.set_title('Heat transport')
ax2.legend(('Total', 'Latent', 'Dry static'))

# ---------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, E_m_P, linewidth=1.5)
ax3.plot(x, Evap, linewidth=1.5)
ax3.plot(x, Prec, linewidth=1.5)
ax3.set_title('E-P, Evap, precip')
ax3.set_xlabel('latitude')
ax3.set_ylabel('m yr^{-1}')
ax3.grid()
ax3.legend(('E - P', 'Evap', 'Precip'))

# ---------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, divFtot)
ax4.plot(x, (1-alb)*S)
ax4.plot(x, A+B*T)
ax4.plot(x, ((1-alb)*S)-(A+B*T)-(divFtot), 'k--', linewidth=1.5)

ax4.set_title('Terms in the energy balance')
ax4.set_xlabel('latitude')
ax4.set_ylabel('W m^{-2}')
ax4.legend(('Transport (\nabla F)', 'net SW', 'net OLR', 'Net'))
ax4.grid()

Tmat = np.squeeze(np.mean(Tout, axis=1))
xvec = np.arange(50)

fig2 = plt.figure(2)
plt.pcolor(xvec, phi, Tmat.T)
cb = plt.colorbar()
