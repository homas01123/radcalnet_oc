import os

import numpy as np
import xarray as xr

import logging

from importlib.resources import files
import yaml

from . import GaseousTransmittance, Misc, LUT, SolarIrradiance

template = files('radcalnet_oc.data.template').joinpath('template.nc')
# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']


class Process():
    def __init__(self,

                 input_db=None,
                 vza=[0],
                 azi=[0],
                 central_wl=np.arange(350, 2500, 1),
                 solar_database='tsis',
                 Rrs_name='Rrs'
                 ):
        '''

        :param input_db:
        :param vza:
        :param azi:
        :param central_wl:
        :param solar_database:
        '''

        if input_db is None:
            input_db = xr.open_dataset(template)

        self.input_db = input_db

        self.vza = vza
        self.azi = azi

        # get LUT object
        logging.info('get LUT object')
        lut = LUT()
        self.lut = lut

        # get full spectral resolution wavelength from gas LUT
        # crop to 350 - 2500 range to comply with OSOAA lut
        full_wl = lut.gas_lut.wl.sel(wl=slice(350, 2500))
        self.full_wl = full_wl
        # lut.wl = full_wl

        logging.info('get auxiliary data')
        lut.load_auxiliary_data()

        logging.info('get solar irradiance')
        solar_irr = SolarIrradiance()
        self.F0 = solar_irr.__dict__[solar_database].interp(wl=full_wl)


        logging.info('get gaseous transmittance object')
        self.gas_trans = GaseousTransmittance(lut.gas_lut)

        logging.info('set input parameters')
        self.set_param(Rrs_name=Rrs_name)

    def set_param(self,
                  Rrs_name='Rrs'):
        '''

        :return:
        '''

        input_db = self.input_db
        self.sza = input_db.sza
        self.mu0 = np.cos(np.radians(self.sza))
        self.muv = np.cos(np.radians(self.vza))

        self.Rrs = input_db[Rrs_name]
        # self.Rrs = self.Rrs.rename({"hyper_wl":"wl"})
        self.aot550 = input_db['aot550']

        # get correction for Sun-Earth distance
        self.D2 = Misc.earth_sun_correction(input_db['day_of_year'])

    def get_gas_transmittance(self):
        '''

        :return:
        '''
        input_db = self.input_db
        self.gas_trans.gas_tc['h2o'] = input_db.tcwv
        self.gas_trans.pressure = input_db['pressure']
        # gas_trans.gas_tc['h2o'] = tcwv
        self.gas_trans.gas_tc['o3'] = input_db['tco3']
        # gas_trans.gas_tc['ch4'] = tcch4
        self.gas_trans.gas_tc['no2'] = input_db['tcno2']

        self.gas_trans.air_mass = 1. / self.mu0
        self.Tg_d = self.gas_trans.get_gaseous_transmittance()

        self.gas_trans.air_mass = 1. / self.muv
        self.Tg_u = self.gas_trans.get_gaseous_transmittance()

    def get_irradiance_transmittance(self):
        '''

        :return:
        '''
        self.Tra_d = self.lut.trans_aero_lut.interp(sza=self.sza,
                                                    aot_ref=self.aot550
                                                    ).interpolate_na('time').interp(wl=self.full_wl, method='quadratic')

    def get_radiance_transmittance(self):
        '''

        :return:
        '''

        tra_u = self.lut.trans_aero_lut.interp(sza=self.vza).interp(aot_ref=self.aot550) ** 1.07
        tra_u = tra_u.interp(wl=self.full_wl, method='quadratic')
        self.tra_u = tra_u.rename({'sza': 'vza'})

    def get_downwelling_irradiance(self,
                                   aerosol_combination=AEROSOL_COMBINATION):
        '''
        Function to get the downwelling irradiance at the bottom-of-atmosphere level.

        :return:
        '''

        self.get_gas_transmittance()

        # atmospheric aerosol-Rayleigh irradiance transmittance
        # self.lut.lut_preparation(sza=self.sza,
        #                         vza=[0],
        #                        aerosol_combination=aerosol_combination)
        self.get_irradiance_transmittance()

        self.Ed = self.Tra_d * self.Tg_d * self.mu0 * self.F0 * self.D2

    def lut_preparation(self,aerosol_combination=AEROSOL_COMBINATION):
        '''

        :param aerosol_combination:
        :return:
        '''
        self.sza_lut = np.sort(np.unique(np.round(self.sza,1)))
        self.lut.lut_preparation(sza=self.sza_lut,
                                 vza=self.vza,
                                 aerosol_combination=aerosol_combination)

    def execute(self,
                aerosol_combination=AEROSOL_COMBINATION
                ):
        '''
        Once all parameters set up, this function proceed with
        the full computation of the top-of-atmosphere exiting radiation for the input time series.

        :return:
        '''

        # atmospheric + sky-reflection radiance
        self.lut.lut_preparation(sza=self.sza,
                                 vza=self.vza,
                                 aerosol_combination=aerosol_combination)

        self.get_gas_transmittance()
        self.get_irradiance_transmittance()
        self.get_radiance_transmittance()

        Ratm = self.lut.Rdiff_lut.squeeze().interp(aot_ref=self.aot550
                                                   ).interp(wl=self.full_wl, method='quadratic')
        Ratm = self.Tg_d * self.Tg_u * Ratm
        self.Ratm = Ratm

        E0 = self.mu0 * self.F0 * self.D2
        self.E0 = E0
        Ed = self.Tra_d * self.Tg_d * E0
        # Ed = Ed.dropna('wl')

        Rrs = self.Rrs.interp(wl=self.full_wl).fillna(0)
        self.Lw_boa =  Rrs * Ed

        Lw_toa = self.tra_u * self.Tg_u * self.Lw_boa
        Lw_toa = Lw_toa.fillna(0)

        Rtoa = np.pi * Lw_toa / E0 + Ratm

        Ed.name = 'Ed'

        radcalnet_db = Ed.reset_coords('sza')
        radcalnet_db['Ed'].attrs = {'unit': 'mW m-2 nm-1',
                                    'description': 'Plane solar irradiance at bottom-of-atmosphere'}

        radcalnet_db['Ratm'] = Ratm.reset_coords(drop=True)
        radcalnet_db['Ratm'].attrs = {'unit': '-',
                                      'description': 'Atmosphere (intrinsic + surface reflected) reflectance at top-of-atmosphere'}

        # radcalnet_db['mu0'] = self.mu0.reset_coords(drop=True)

        radcalnet_db['Rrs'] = Rrs.reset_coords(drop=True)
        radcalnet_db['Rrs'].attrs = {'unit': '-',
                                     'description': 'Remote sensing reflectance',
                                     'source': 'from AERONET-OC and spectral interpolation based on InvRrs'}

        radcalnet_db['E0'] = E0.reset_coords(drop=True)
        radcalnet_db['E0'].attrs = {'unit': 'mW m-2 nm-1',
                                    'description': 'Plane solar irradiance at top-of-atmosphere'}

        #radcalnet_db['Lw_toa'] = Lw_toa.reset_coords(drop=True)
        #radcalnet_db['Lw_toa'].attrs = {'unit': 'mW m-2 sr-1 nm-1',
        #                                'description': 'water-leaving radiance at top-of-atmosphere'}

        radcalnet_db['Rtoa'] = Rtoa.reset_coords(drop=True)
        radcalnet_db['Rtoa'].attrs = {'unit': '-',
                                      'description': 'Total reflectance at top-of-atmosphere'}

        radcalnet_db = xr.merge([radcalnet_db,
                                 self.input_db[['tcwv', 'tco3', 'tcno2', 'pressure',
                                                'aot550', 'aot865', 'ang_exp']]])

        radcalnet_db['aerosol_combination'] = self.lut.aerosol_combination
        radcalnet_db['aerosol_combination'].attrs = {'description':
                                                        'relative proportion of each aerosol model used for the computation',
                                                    'models': AEROSOL_MODELS }
        radcalnet_db['D2'] = self.D2

        self.radcalnet_db = radcalnet_db.squeeze()

