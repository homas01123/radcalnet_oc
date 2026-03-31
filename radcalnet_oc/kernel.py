

import os, sys
import numpy as np
import xarray as xr

from scipy.optimize import least_squares

import yaml
from importlib.resources import files
import logging

opj = os.path.join


# ------------------------------------
# get path of packaged files
# ------------------------------------
LUT_FILE_BACKUP = files('radcalnet_oc.data.lut.atmo').joinpath('toa_lut_opac_ultra_light.nc')

# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

LUTDATA = config['path']['lutdata']
TOALUT = config['path']['toa_lut']
AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']
NETCDF_ENGINE = config['processor']['netcdf_engine']

class Aerosol:

    def __init__(self,
                 naot_db,
                 naot_lut,
                 wl_dimension="wl_aeronet",
                 ):
        '''
        Algorithm to retrieve aerosol models mixture from spectral aerosol optical thickness
        The algorithm is based on the aerosol models used to  build the variable self.naot_lut
        
        :param naot_db: xarray with "time" dimension of normalized aerosol optical thickness 
        :param wl_dimension: name of the spectral dimension of the input naot_db
        '''

        self.naot_db = naot_db.rename({wl_dimension: "wl"})
        self.wl = self.naot_db.wl

        # replace NaN for further least_square optimization
        self.naot_db = self.naot_db.interpolate_na('wl')
        if "time" in self.naot_db.dims:
            self.naot_db = self.naot_db.interpolate_na('time')

        # get normalized aot for aerosol model fitting
        self.naot_lut = naot_lut
        self.naot_lut_all = naot_lut_all = self.naot_lut.interp(wl=self.wl, method='quadratic')

        # naot_lut is used to fit the proportion of each model
        self.naot_lut = np.array([naot_lut_all.sel(model="DESE_rh70"),
                                  naot_lut_all.sel(model="MACL_rh70"),
                                  naot_lut_all.sel(model="WASO_rh0")])


    def func_aero(self, fcoef, n_aot):
        '''function to fit spectral behavior of bimodal aerosols
         onto aeronet optical thickness'''
        fcoef = fcoef / np.sum(fcoef)
        sim = fcoef[0] * n_aot[0] + fcoef[1] * n_aot[1] + fcoef[2] * n_aot[2]
        return sim

    def cost_func(self,
                  fcoef,
                  n_aot_lut,
                  naot_mes):
        '''
        Cost function for aerosl model retrieval
        :param fcoef: proportions of each model
        :param n_aot_lut: LUT with the spectral aot of each model
        :param naot_mes: spectral aot from measurements
        :return:
        '''
        sim = self.func_aero(fcoef, n_aot_lut)
        return naot_mes - sim

    def process(self
                ):
        '''
        Optimization process to retrieve the proportion of each selected aerosol models

        :return:
        '''


        xarr = []
        for time, naot in self.naot_db.groupby('time'):
            naot = naot.squeeze()
            p0 = [0., 0.5, 0.5]
            res = least_squares(self.cost_func, p0,
                                args=(self.naot_lut, naot),
                                bounds=([0, 0, 0], [1, 1, 1]))
            res.x = res.x / np.sum(res.x)
            # results are time, the proportion of each aerosol model, array ends with the remaining cost
            xarr.append([time, 0, 0, res.x[0], res.x[1], 0, res.x[2], res.cost])
            # xarr.append([time,0,0,0,1,0,0,res.cost])
        xarr = np.array(xarr)

        self.model_db = xr.Dataset(data_vars=dict(aerosol_combination=(['time', 'model'], xarr[:, 1:-1].astype(float)),
                                          cost=('time', xarr[:, -1].astype(float)),
                                          ),
                           coords={'time': xarr[:, 0],
                                   'model': AEROSOL_MODELS})


class CamsParams:
    def __init__(self,
                 name,
                 resol):
        self.name = name
        self.resol = resol


class Gases():

    def __init__(self):
        '''
        Intermediate class to set parameters for absorbing gases.
        '''

        self.pressure = 1010
        self.pressure_gas_ref = 1000

        self.gas_tc = {'co2': 1,
                       'o2': 1,
                       'o4': 1,
                       'ch4': 1e-2,
                       'no2': 3e-6,
                       'o3': 6.5e-3,
                       'h2o': 30}
        self.coef_abs_scat = {'co2': 1.,
                              'o2': 1.,
                              'o4': 1.,
                              'ch4': 1.,
                              'no2': 1,
                              'o3': 1,
                              'h2o': 1.}


class GaseousTransmittance(Gases):

    def __init__(self,
                 gas_lut: xr.DataArray,
                 zenith_angle=0
                 ):
        '''
        Class containing functions to compute the direct transmittance of the absorbing gases.
        :param gas_lut: xarray.DataArray Look-up table data for gaseous absorption
        :param zenith_angle: zenith angle (solar or viewing) in degrees
        '''
        Gases.__init__(self)
        self.air_mass = np.cos(np.radians(zenith_angle))
        self.gas_lut = gas_lut

    def Tgas_background(self):
        '''
        Compute direct transmittance for background absorbing gases: :math:`CO, O_2, O_4`

        :return:
        '''
        gl = self.gas_lut
        self.ot_air = self.pressure / self.pressure_gas_ref * \
                      (gl.co + self.coef_abs_scat['co2'] * gl.co2 +
                       self.coef_abs_scat['o2'] * gl.o2 +
                       self.coef_abs_scat['o4'] * gl.o4)
        self.Tg_bg = np.exp(- self.air_mass * self.ot_air)
        return self.Tg_bg

    def Tgas(self,
             gas_name,
             ):
        '''
        Compute hyperspectral transmittance for a given absorbing gas and
        convolve it with the spectral response functions of the satellite sensor.

        :param gas_name: name of the absorbing gas, choose between:
            - 'h2o'
            - 'o3'
            - 'n2o'
        :return: Gaseous transmittance for satellite bands
        '''

        ot = self.coef_abs_scat[gas_name] * self.gas_tc[gas_name] * self.gas_lut[gas_name]
        Tg = np.exp(- self.air_mass * ot)
        return Tg

    def get_gaseous_transmittance(self,
                                  gases=['ch4', 'no2', 'o3', 'h2o'],
                                  background=True):
        '''
        Get the final total gaseous transmittance.
        :return:
        '''

        first = True
        for gas_name in gases:
            if first:
                Tg_tot = self.Tgas(gas_name)
                first = False
            else:
                Tg_tot = Tg_tot * self.Tgas(gas_name)

        if background:
            Tg_tot = Tg_tot * self.Tgas_background()

        return Tg_tot


class Misc:
    '''
    Miscelaneous utilities
    '''

    @staticmethod
    def get_pressure(alt, psl):
        '''Compute the pressure for a given altitude
           alt : altitude in meters (float or np.array)
           psl : pressure at sea level in hPa
           palt : pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))

    @staticmethod
    def earth_sun_correction(dayofyear):
        '''
        Earth-Sun distance correction factor for adjustment of mean solar irradiance

        :param dayofyear:
        :return: correction factor
        '''
        theta = 2. * np.pi * dayofyear / 365
        d2 = 1.00011 + 0.034221 * np.cos(theta) + 0.00128 * np.sin(theta) + \
             0.000719 * np.cos(2 * theta) + 0.000077 * np.sin(2 * theta)
        return d2