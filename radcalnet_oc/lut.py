import os

import numpy as np
import xarray as xr
import pandas as pd
import datetime

from numba import njit, prange

import matplotlib.pyplot as plt

import logging
from importlib.resources import files
import yaml

opj = os.path.join

# ------------------------------------
# get path of packaged files
# ------------------------------------
dir, filename = os.path.split(__file__)

thuillier_file = files('radcalnet_oc.data.auxdata').joinpath('ref_atlas_thuillier3.nc')
gueymard_file = files('radcalnet_oc.data.auxdata').joinpath('NewGuey2003.dat')
kurucz_file = files('radcalnet_oc.data.auxdata').joinpath('kurucz_0.1nm.dat')
tsis_file = files('radcalnet_oc.data.auxdata').joinpath(
    'hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc')
sunglint_eps_file = files('radcalnet_oc.data.auxdata').joinpath('mean_rglint_small_angles_vza_le_12_sza_le_60.txt')
rayleigh_file = files('radcalnet_oc.data.auxdata').joinpath('rayleigh_bodhaine.txt')
LUT_FILE_BACKUP = files('radcalnet_oc.data.lut.atmo').joinpath('toa_lut_opac_ultra_light.nc')

# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

LUTDATA = config['path']['lutdata']
TOALUT = config['path']['toa_lut']
TRANSLUT = config['path']['trans_lut']
TRANSLUT = files('radcalnet_oc.data.lut.atmo').joinpath(TRANSLUT)
CAMS_PATH = config['path']['trans_lut']
NCPU = config['processor']['ncpu']
NETCDF_ENGINE = config['processor']['netcdf_engine']

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']


@njit(fastmath=True)
def Gamma2sigma(Gamma):
    '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
    return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)


@njit(parallel=True, fastmath=True)
def gaussian(x, mu, sigma):
    '''
    Generate gaussian distribution
    :param x:
    :param mu: mode of the Gaussian distribution
    :param sigma: Standard deviation of the Gaussian distribution
    :return:
    '''
    result = np.full((len(x)), np.nan, dtype=np.float32)
    for i in prange(len(result)):
        result[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x[i] - mu) ** 2 / (2 * sigma ** 2))
    return result


@njit(fastmath=True)
def super_gaussian(x,
                   amplitude=1.0,
                   mu=0.0,
                   sigma=1.0,
                   expon=2.0):
    '''
    Super-Gaussian distribution:
    super_gaussian(x, amplitude, mu, sigma, expon) =
        (amplitude/(sqrt(2*pi)*sigma)) * exp(-abs(x-mu)**expon / (2*sigma**expon))
    :param x:
    :param amplitude:
    :param mu:
    :param sigma:
    :param expon:
    :return:
    '''

    sigma = max(1.e-15, sigma)
    return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))


@njit(fastmath=True)
def super_gaussian_fwhm2sigma(fwhm,
                              expon):
    '''
    Function to convert FWHM to standard deviation (sigma) of the super-gaussian distribution
    :param fwhm:
    :param expon:
    :return:
    '''
    return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)


class LUT:
    def __init__(self,
                 wl=np.arange(350, 2500, 10),
                 lut_file=opj(LUTDATA, TOALUT),
                 trans_lut_file=TRANSLUT):
        '''
        Module to load LUT files.
        :param wl: array of wavelength to process in nm
        :param lut_file: path for diffuse light radiation LUT
        :param trans_lut_file: path for irradiance transmittance LUT
        '''

        # set parameters
        self.wl = wl

        # get path of necessary look-up tables

        self.lut_file = lut_file
        self.trans_lut_file = trans_lut_file
        self.dirdata = config['path']['lutdata']
        self.abs_gas_file = files('radcalnet_oc.data.lut.gases') / 'lut_abs_opt_thickness_normalized.nc'
        # self.lut_file = opj(self.dirdata, 'lut', 'opac_osoaa_lut_v2.nc')
        self.water_vapor_transmittance_file = files('radcalnet_oc.data.lut.gases') / 'water_vapor_transmittance.nc'

        self.load_auxiliary_data()

    def load_auxiliary_data(self):
        '''
        Load look-up tables data for gas absorption and backgroud transmittance

        :return:
        '''

        logging.info('loading look-up tables')
        self.trans_lut = xr.open_dataset(self.trans_lut_file, engine=NETCDF_ENGINE)


        # convert wavelength in nanometer
        self.trans_lut['wl'] = self.trans_lut['wl'] * 1000
        self.trans_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        try:
            self.aero_lut = xr.open_dataset(self.lut_file, engine=NETCDF_ENGINE)
        except:
            logging.info('LUT file ' + self.lut_file + ' not found, please download and save it the proper directory')
            logging.info('LUT has been replaced with light dataset that might produce inaccuracies')
            self.aero_lut = xr.open_dataset(LUT_FILE_BACKUP, engine=NETCDF_ENGINE)

        # convert wavelength in nanometer
        self.aero_lut['wl'] = self.aero_lut['wl'] * 1000
        self.aero_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        self.aero_lut['aot'] = self.aero_lut.aot.isel(wind=0).squeeze()

        # get normalized aot for aerosol model fitting
        self.naot_lut = self.aero_lut.aot.interp(aot_ref=0.1) / 0.1

        self.gas_lut = xr.open_dataset(self.abs_gas_file, engine=NETCDF_ENGINE)
        self.Twv_lut = xr.open_dataset(self.water_vapor_transmittance_file, engine=NETCDF_ENGINE)

    def lut_preparation(self,
                        wind=2,
                        sza=[20, 40, 60],
                        vza=[0],
                        azi=[0],
                        aot_refs=np.linspace(0.0, 0.8, 25),
                        aerosol_combination=AEROSOL_COMBINATION):

        logging.info('LUT preparation')

        if isinstance(aerosol_combination, (list, np.ndarray)):
            aerosol_combination = xr.DataArray(aerosol_combination, coords={'model': AEROSOL_MODELS})
        elif not isinstance(aerosol_combination, xr.DataArray):
            logging.error("error in aerosol_combination parameter, should be numpy or xr.DataArray")
            return

        self.aerosol_combination=aerosol_combination

        aero_lut = self.aero_lut.sel(wind=wind, method='nearest')
        trans_lut = self.trans_lut.sel(wind=wind, method='nearest')

        # -----------------------------
        # interpolation transmittance
        # -----------------------------
        self.trans_aero_lut = trans_lut.interp(sza=[*sza, *vza]).sortby('sza')
        self.trans_aero_lut = (self.trans_aero_lut * aerosol_combination).sum('model')
        self.trans_aero_lut = self.trans_aero_lut.interp(aot_ref=aot_refs, method='quadratic')
        #self.trans_aero_lut = self.trans_aero_lut.interp(wl=self.wl, method='quadratic')
        # clean up the xarray DataArray object:
        self.trans_aero_lut = self.trans_aero_lut.to_dataarray().squeeze().reset_coords(drop=True)

        # -----------------------------
        # interpolation Rayleigh
        # -----------------------------
        self.Rray = aero_lut.I.isel(model=0).interp(sza=sza, vza=vza).interp(azi=azi).interp(aot_ref=0,
                                                                                             method='quadratic')
        self.Rray = self.Rray / np.cos(np.radians(self.Rray.sza))
        #self.Rray = self.Rray.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation atmo diffuse light
        # -----------------------------
        self.Rdiff_lut = aero_lut.I.interp(sza=sza, vza=vza).interp(azi=azi)
        self.Rdiff_lut = (self.Rdiff_lut * aerosol_combination).sum('model')
        self.Rdiff_lut = self.Rdiff_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.Rdiff_lut = self.Rdiff_lut / np.cos(np.radians(self.Rdiff_lut.sza))
        #self.Rdiff_lut = self.Rdiff_lut.interp(wl=self.wl, method='quadratic')

        self.aot_lut = (aero_lut.aot * aerosol_combination).sum('model')
        self.aot_lut = self.aot_lut.interp(aot_ref=aot_refs,
                                           method='quadratic'
                                           )#.interp(wl=self.wl, method='quadratic')

        self.szas = self.Rdiff_lut.sza.values
        self.vzas = self.Rdiff_lut.vza.values
        self.azis = self.Rdiff_lut.azi.values
        self.aot_refs = self.Rdiff_lut.aot_ref.values

        _auxdata = AuxData(wl=self.wl)  # wl=masked.wl)
        self.sunglint_eps = _auxdata.sunglint_eps  # ['mean'].interp(wl=wl)
        self.rot = _auxdata.rot

    def lut_preparation_all_models(self,
                                   wind=2,
                                   sza=[20, 40, 60],
                                   vza=[0],
                                   azi=[0],
                                   aot_refs=np.linspace(0.0, 0.8, 25),
                                   ):

        logging.info('LUT preparation')

        aero_lut = self.aero_lut.sel(wind=wind, method='nearest')
        trans_lut = self.trans_lut.sel(wind=wind, method='nearest')

        # -----------------------------
        # interpolation transmittance
        # -----------------------------
        self.trans_aero_lut = trans_lut.interp(sza=[*sza, *vza])
        self.trans_aero_lut = self.trans_aero_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.trans_aero_lut = self.trans_aero_lut.interp(wl=self.wl, method='quadratic')
        # clean up the xarray DataArray object:
        self.trans_aero_lut = self.trans_aero_lut.to_dataarray().squeeze().reset_coords(drop=True)

        # -----------------------------
        # interpolation Rayleigh
        # -----------------------------
        self.Rray = aero_lut.I.interp(sza=sza, vza=vza).interp(azi=azi).interp(aot_ref=0, method='quadratic')
        self.Rray = self.Rray / np.cos(np.radians(self.Rray.sza))
        self.Rray = self.Rray.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation atmo diffuse light
        # -----------------------------
        self.Rdiff_lut = aero_lut.I.interp(sza=sza, vza=vza)
        self.Rdiff_lut = self.Rdiff_lut.interp(azi=azi).interp(aot_ref=aot_refs, method='quadratic')
        self.Rdiff_lut = self.Rdiff_lut / np.cos(np.radians(self.Rdiff_lut.sza))
        self.Rdiff_lut = self.Rdiff_lut.interp(wl=self.wl, method='quadratic')

        self.aot_lut = aero_lut.aot.interp(aot_ref=aot_refs, method='quadratic').interp(wl=self.wl, method='quadratic')

        self.szas = self.Rdiff_lut.sza.values
        self.vzas = self.Rdiff_lut.vza.values
        self.azis = self.Rdiff_lut.azi.values
        self.aot_refs = self.Rdiff_lut.aot_ref.values

        _auxdata = AuxData(wl=self.wl)  # wl=masked.wl)
        self.sunglint_eps = _auxdata.sunglint_eps  # ['mean'].interp(wl=wl)
        self.rot = _auxdata.rot


class AuxData():
    def __init__(self,
                 wl=None):
        # load data from raw files
        # self.solar_irr = SolarIrradiance()
        self.sunglint_eps = pd.read_csv(sunglint_eps_file, sep=r'\s+', index_col=0).to_xarray()
        self.rayleigh()
        self.pressure_rot_ref = 1013.25

        # reproject onto desired wavelengths
        if wl is not None:
            # self.solar_irr = self.solar_irr.interp(wl=wl)
            self.sunglint_eps = self.sunglint_eps['mean'].interp(wl=wl)
            self.rot = self.rot.interp(wl=wl)

    def rayleigh(self):
        '''
        Rayleigh Optical Thickness for
        P=1013.25mb,
        T=288.15K,
        CO2=360ppm
        from
        Bodhaine, B.A., Wood, N.B, Dutton, E.G., Slusser, J.R. (1999). On Rayleigh
        Optical Depth Calculations, J. Atmos. Ocean Tech., 16, 1854-1861.
        '''
        data = pd.read_csv(rayleigh_file, skiprows=16, sep=' ', header=None)
        data.columns = ('wl', 'rot', 'dpol')
        self.rot = data.set_index('wl').to_xarray().rot


class SolarIrradiance():
    def __init__(self, wl=None):
        # load data from raw files
        self.wl_min = 300
        self.wl_max = 2600

        self.gueymard = self.read_gueymard()
        self.kurucz = self.read_kurucz()
        self.thuillier = self.read_thuillier()
        self.tsis = self.read_tsis()

    def read_tsis(self):
        '''
        Open TSIS data and convert them into xarray in mW/m2/nm
        :return:
        '''
        tsis = xr.open_dataset(tsis_file)
        tsis = tsis.set_index(wavelength='Vacuum Wavelength').rename(
            {'wavelength': 'wl'})  # set_coords('Vacuum Wavelength')
        # convert
        tsis['SSI'] = tsis.SSI * 1000  # .plot(lw=0.5)
        tsis.SSI.attrs['units'] = 'mW m-2 nm-1'
        tsis.SSI.attrs['long_name'] = 'Solar Spectral Irradiance Reference Spectrum (mW m-2 nm-1)'
        tsis.SSI.attrs['reference'] = 'Coddington, O. M., Richard, E. C., Harber, D., et al. (2021).' + \
                                      'The TSIS-1 Hybrid Solar Reference Spectrum. Geophysical Research Letters,' + \
                                      '48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709'
        return tsis.SSI.sel(wl=slice(self.wl_min, self.wl_max))

    def read_thuillier(self):
        '''
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
        '''
        solar_irr = xr.open_dataset(thuillier_file).squeeze().data.drop('time') * 1e3
        solar_irr = solar_irr.rename({'wavelength': 'wl'})
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        return solar_irr

    def read_gueymard(self):
        '''
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
        '''
        solar_irr = pd.read_csv(gueymard_file, sep=r'\s+', skiprows=30, header=None)
        solar_irr.columns = ['wl', 'data']
        solar_irr = solar_irr.set_index('wl').data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        solar_irr.attrs['reference'] = 'Gueymard, C. A., Solar Energy, Volume 76, Issue 4,2004, ISSN 0038-092X'
        return solar_irr

    def read_kurucz(self):
        '''
        Open Kurucz data and convert them into xarray in mW/m2/nm
        :return:
        '''
        solar_irr = pd.read_csv(kurucz_file, sep=r'\s+', skiprows=11, header=None)
        solar_irr.columns = ['wl', 'data']
        solar_irr = solar_irr.set_index('wl').data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        solar_irr.attrs['reference'] = 'Kurucz, R.L., Synthetic infrared spectra, in Infrared Solar Physics, ' + \
                                       'IAU Symp. 154, edited by D.M. Rabin and J.T. Jefferies, Kluwer, Acad., ' + \
                                       'Norwell, MA, 1992.'
        return solar_irr

    def interp(self, wl=[440, 550, 660, 770, 880]):
        '''
        Interpolation on new wavelengths
        :param wl: wavelength in nm
        :return: update variables of the class
        '''
        self.thuillier = self.thuillier.interp(wl=wl)
        self.gueymard = self.gueymard.interp(wl=wl)


class Spectral():
    def __init__(self,
                 central_wl,
                 fwhm):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param central_wl: numpy array of the central wavelengths
        :param fwhm: scalar or numpy array containing full width at half maximum in nm                :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''
        self.central_wl = central_wl
        if not isinstance(fwhm, np.ndarray):
            fwhm = np.array([fwhm] * len(central_wl))
        fwhm = xr.DataArray(fwhm, name='fwhm',
                            coords={'wl': central_wl},
                            attrs={
                                'definition': 'full width at half maximum of spectral responses modeled as gaussian distributions'})
        self.fwhm = fwhm

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.fwhm.groupby('wl'):
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

        return fig

    @staticmethod
    @njit(parallel=True)
    def convolve_(
            wl_signal,
            signal,
            wl,
            fwhm,
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :return: numpy of convoluted signal
        '''
        Nwl = len(wl)
        signal_ = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = Gamma2sigma(fwhm[ii])
            rsr = gaussian(wl_signal, wl[ii], sig)
            signal_[ii] = np.trapezoid((signal * rsr), wl_signal) / np.trapezoid(rsr, wl_signal)
        return signal_

    @staticmethod
    @njit(parallel=True)
    def convolve2_(
            wl_signal,
            signal,
            wl,
            fwhm,
            expon=2.,
            threshold=1e-6
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :param threshold: minimum values of the response function to be included in the convolution
        :return: numpy of convoluted signal
        '''

        Nwl = len(wl)
        response = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = super_gaussian_fwhm2sigma(fwhm[ii], expon)
            rsr = super_gaussian(wl_signal, mu=wl[ii], sigma=sig, expon=expon)

            # remove values above a given threshold to speed up computation
            idx = rsr > threshold
            wl_signal_ =wl_signal[idx]
            signal_ = signal[idx]
            rsr = rsr[idx]

            response[ii] = np.trapezoid((signal_ * rsr), wl_signal_) / np.trapezoid(rsr, wl_signal_)
        return response

    def convolve2(self,
                  signal,
                  name='signal',
                  expon=3,
                  threshold=1e-4,
                  info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :param threshold: minimum values of the response function to be included in the convolution
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims
        attrs=signal.attrs
        name=signal.name
        if len(xdims) == 1:
            signal_int = self.convolve2_(wl_ref, signal.values, wl, fwhm, expon, threshold=threshold)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=attrs)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve2_(signal_.wl.values, signal_.values, wl, fwhm, expon)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int)#.to_dataarray()
            signal_int.attrs = attrs

        return signal_int

    def convolve(self,
                 signal,
                 name='signal',
                 info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims

        if len(xdims) == 1:
            signal_int = self.convolve_(wl_ref, signal.values, wl, fwhm)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=info)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve_(signal_.wl.values, signal_.values, wl, fwhm)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int).to_dataarray()
            signal_int.attrs = info

        return signal_int
