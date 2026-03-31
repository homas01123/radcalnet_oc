import os
import pandas as pd
import numpy as np
import re
import datetime as dt
from scipy.interpolate import interp1d
from importlib.resources import files

# ------------------------------------
# get path of packaged files
# ------------------------------------
dir, filename = os.path.split(__file__)

aeronet_oc_location_file = files('radcalnet_oc.data.aeronet'
                                 ).joinpath('info_aeronet_oc_sentinel2_location.csv')


class Aeronet:
    def __init__(self):
        self.get_aeronet_oc_site_info()

    def get_aeronet_oc_site_info(self):
        self.aeronet_oc_site_info = pd.read_csv(aeronet_oc_location_file, index_col=0)



    def read_aeronet_ocv3(self,
                          ifile,
                          skiprows=8,
                          encoding='latin_1'):
        '''
         Read and format in pandas data.frame the standard AERONET-OC data

        :param skiprows: number of rows to skip in the ASCII AERONET file
        :return:
        '''

        h1 = pd.read_csv(ifile, skiprows=skiprows - 1, nrows=1, encoding=encoding).columns[3:]
        h1 = np.insert(h1, 0, 'site')
        data_type = h1.str.replace(r'\[.*\]', '', regex=True)
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength', regex=True)

        # convert into float to order the dataframe with increasing wavelength
        h2 = h1.str.replace(r'.*\[', '', regex=True)
        h2 = h2.str.replace(r'nm\].*', '', regex=True)
        h2 = h2.str.replace(r'Exact_Wavelengths\(um\)_', '', regex=True)
        h2 = pd.to_numeric(h2, errors='coerce')  # h2.str.extract('(\d+)').astype('float')
        h2 = h2.fillna('').T

        df = pd.read_csv(ifile, skiprows=skiprows, header=None,
                         na_values=['N/A', -999.0, -9.999999], index_col=False, encoding=encoding)

        df['time'] = pd.to_datetime(df.pop(1).astype(str) + df.pop(2).astype(str), format="%d:%m:%Y%H:%M:%S")
        # df['site'] = site
        # df.set_index(['site', 'time'],inplace=True)
        df.set_index('time', inplace=True)

        tuples = list(zip(h1, data_type, h2))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in df.columns])
        df.sort_index(axis=1, level=2, inplace=True)
        return df

    def read_aeronet(self,
                     skiprows=6,
                     encoding='latin_1'):
        ''' Read and format in pandas data.frame the V3 AERONET data '''

        ifile = self.file
        df = pd.read_csv(ifile, skiprows=skiprows, nrows=1, encoding=encoding)  # read just first line for columns
        columns = df.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]  # drop the last one
        df = pd.read_csv(ifile,
                         skiprows=skiprows,
                         usecols=cols_to_use,
                         index_col=False,
                         na_values=['N/A', -999.0],
                         encoding=encoding)

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.rename(columns={'AERONET_Site_Name': 'site', 'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'},
                  inplace=True)
        format = "%d:%m:%Y%H:%M:%S"
        df['time'] = pd.to_datetime(df[df.columns[0]] + df[df.columns[1]], format=format)
        # df.set_index(['site','time'], inplace=True)
        df.set_index('time', inplace=True)
        df = df.drop(df.columns[[0, 1]], axis=1)
        # df['year'] = df.index.get_level_values(1).year

        # cleaning up
        df.drop(list(df.filter(regex='Input')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Empty')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Day')), axis=1, inplace=True)

        # indexing columns with spectral values
        data_type = df.columns.str.replace(r'AOD.*nm', 'aot', regex=True)
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength', regex=True)
        data_type = data_type.str.replace(r'Triplet.*[0-9]', 'std', regex=True)
        data_type = data_type.str.replace(r'^(?!aot|std|wavelength).*$', '', regex=True)

        wl_type = df.columns.str.extract(r'(\d+)').astype('float')
        wl_type = wl_type.fillna('')

        tuples = list(zip(df.columns, data_type, wl_type))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        if 'wavelength' in df.columns.levels[1]:
            df.loc[:, (slice(None), 'wavelength',)] = df.loc[:, (slice(None), 'wavelength')] * 1000  # convert into nm
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.sort_index(axis=1, level=2, inplace=True)
        return df

    def read_aeronet_inv(self,
                         skiprows=6,
                         encoding='latin_1'):
        ''' Read and format in pandas data.frame the V3 Aerosol Inversion AERONET data '''
        ifile = self.file
        df = pd.read_csv(ifile, skiprows=skiprows, nrows=1,encoding=encoding)  # read just first line for columns
        columns = df.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]  # drop the last one
        df = pd.read_csv(ifile,
                         skiprows=skiprows,
                         usecols=cols_to_use,
                         index_col=False,
                         na_values=['N/A', -999.0],
                         encoding=encoding)

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.rename(columns={'AERONET_Site_Name': 'site', 'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date', },
                  inplace=True)
        format = "%d:%m:%Y %H:%M:%S"
        df['time'] = pd.to_datetime(df[df.columns[1]] + ' ' + df[df.columns[2]], format=format)
        # df.set_index(['site','time'], inplace=True)
        df.set_index('time', inplace=True)
        df = df.drop(df.columns[[0, 1]], axis=1)
        # df['year'] = df.index.get_level_values(1).year

        # cleaning up
        df.drop(list(df.filter(regex='Input')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Empty')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Day')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Angle_Bin')), axis=1, inplace=True)

        # indexing columns with spectral values
        data_type = df.columns.str.replace(r'AOD.*nm', 'aot')
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength')
        data_type = data_type.str.replace(r'Triplet.*[0-9]', 'std')
        data_type = data_type.str.replace(r'^(?!aot|std|wavelength).*$', '')

        wl_type = df.columns.str.extract(r'(\d+)').astype('float')
        wl_type = wl_type.fillna('')

        tuples = list(zip(df.columns, data_type, wl_type))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        if 'wavelength' in df.columns.levels[1]:
            df.loc[:, (slice(None), 'wavelength',)] = df.loc[:, (slice(None), 'wavelength')] * 1000  # convert into nm

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.sort_index(axis=1, level=2, inplace=True)

        return df
