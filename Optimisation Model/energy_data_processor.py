#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import pvlib
from pvlib import pvsystem
import datetime as dt



class EnergyDataProcessor:
    """
    Class to handle data processing for energy optimization.

    Attributes:
        year (int): The year of assessment.
        lat (float): self.lat of the location.
        lon (float): self.lon of the location.
    """
    
    def __init__(self, year, lat, lon):
        """
        Initializes the EnergyDataProcessor class.

        Parameters:
            year (int): The year of assessment.
            lat (float): self.lat of the location.
            lon (float): self.lon of the location.
        """
        self.year = year
        self.lat = lat
        self.lon = lon


    def get_weather_data(self, start_year=2013, end_year=2023):
        """
        Gets location-specific weather data from PVGIS.

        Args:
            self.lat (float): self.lat [degrees].
            self.lon (float): self.lon [degrees].
            start_year (int): start of typical meteorogical year (TMY) data, std is 2007 [int].
            end_year (int): start of typical meteorogical year (TMY) data, std is 2016 [int].
        Returns:
            data: dataframe with annual data based on typical meteorogical year data of PVGIS, with:
                    Date & time (UTC for normal CSV, local timezone time for the EPW format)
                    temp_air - Dry bulb (air) temperature.
                    relative_humidity [%] - Relative Humidity.
                    ghi [W/m2] - Global horizontal irradiance.
                    dni [W/m2] - Direct (beam) irradiance.
                    dhi [W/m2] - Diffuse horizontal irradiance.
                    IR(h) [W/m2] - Infrared radiation downwards.
                    wind_speed [m/s] - Windspeed.
                    wind_direction [°] - Wind direction.
                    pressure [Pa] - Surface (air) pressure.
            elevation: elevation of location [meters]
        """

        data, metadata = pvlib.iotools.get_pvgis_tmy(float(self.lat), float(self.lon), startyear=start_year, endyear=end_year, 
                                         outputformat='csv')
        elevation = metadata["inputs"]["elevation"]
        data = data.reset_index()
        data['time(UTC)'] = data['time(UTC)'].astype(str)

        data['time(UTC)'] = data['time(UTC)'].str[:-9]#.astype(float)
        data['time(UTC)'] = data['time(UTC)'].apply(lambda x: str(self.year) + x[4:])
        data['time(UTC)'] = data['time(UTC)'].apply(lambda x: dt.datetime.strptime(str(x),'%Y-%m-%d %H:%M'))  

        data = data.set_index('time(UTC)')

        data['ghi'] = data['ghi'].fillna(0)
        data['dni'] = data['dni'].fillna(0)
        data['dhi'] = data['dhi'].fillna(0)

        data['wind_speed'][data['wind_speed']<0] = 0

        return data, elevation

    def get_renewable_profiles(self, weather_data):
        """
        Calculates renewable electricity generation profiles for solar PV, onshore wind, and offshore wind for TMY per kWp.

        Args:
            weather_data (dataframe): dataframe with weather information for one year [pd.Dataframe].
            self.lat (float): self.lat, in decimal degrees, between -90 and 90, north is positive (ISO 19115) [degrees].
            self.lon (float): self.lon, in decimal degrees, between -180 and 180, east is positive (ISO 19115) [degrees].
        Returns:
            pv_MW (np.array): the pv profile per kWp module modelled [kWp].
            wind_MW_on (np.array): the wind profile for on-shore per kWp module modelled [kWp].
            wind_MW_off (np.array): the wind profile for off-shore per kWp module modelled [kWp] [kWp].
        """

        #PV
        pv_MW_start = pv_profile_generator_tmy(weather_data, self.lat, self.lon)
        pv_MW = np.array(pv_MW_start)


        print("pv cf '{}'".format(round(pv_MW.mean(),3)))
        return pv_MW



def pv_profile_generator_tmy(weather_data, latitude: float, longitude: float,
                             surface_azimuth=0, surface_tilt=35,
                             module_spec='Sharp_NDQ235F4__2013_',
                             type_pv='roof_mounted'):
    """
    Calculates PV load factor profile (kWp generated per kWp installed).

    Parameters
    ----------
    weather_data: pandas.DataFrame
        Dataframe with weather information for one year [pd.DataFrame].

    latitude: float
        Latitude, in decimal degrees, between -90 and 90, north is positive
        (ISO 19115) [degrees].

    longitude: float
        Longitude, in decimal degrees, between -180 and 180, east is positive
        (ISO 19115) [degrees].

    surface_azimuth: float
        Panel orientation surface azimuth angle, in decimal degrees, between
        0 and 359.99. Default is 0 (north) as in Southern Hemispere [degrees].

    surface_tilt: float
        Panel tilt from horizontal surface angle, in decimal degrees, between
        0 and 90. Default is 35 as is generally optimal [degrees].

    module_spec: str
        Type of solar module to be modelled [str] default is
        Sharp_NDQ235F4__2013_. The PV module modelled, here standard the Sharp,
        since parameters are available and multi-Si

    type_pv: str
        How the module will be installed. Default is roof_mounted.

    Returns
    -------
    total_profile_tmy: pandas.Series
        The PV load factor profile, for a TMY [kWp generated/kWp install].
    """

    # If TMY, delete first substrings from rows to avoid inconsitencies
    temperature = weather_data['temp_air'].astype(float)
    windspeed = weather_data['wind_speed'].astype(float)
    ghi = weather_data['ghi']
    dni = weather_data['dni']
    dhi = weather_data['dhi']
    pressure = weather_data['pressure'].astype(float)

    # Solar details for location
    sp = pvlib.solarposition.get_solarposition(weather_data.index,
                                               float(latitude),
                                               float(longitude),
                                               pressure=pressure
                                               )
    irradiance = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                       surface_azimuth,
                                                       sp.apparent_zenith,
                                                       sp.azimuth,
                                                       dni, ghi, dhi
                                                       )

    # Get db with modules, choose the one in the function and get specs
    modules = pvsystem.retrieve_sam('SandiaMod').T
    modules_multi_si = modules[modules['Material'] == 'mc-Si']
    multi_si = modules_multi_si[
        modules_multi_si.index.str.contains(module_spec)].T
    multi_si = multi_si[module_spec]

    # Relative airmass and aoi
    airmass = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith)
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass,
                                                             pressure=pressure
                                                             )

    # Angle of incidence
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               sp.zenith, sp.azimuth
                               )

    # Cell temperature
    if type_pv == 'roof_mounted':
        temp_model_params = (pvlib.temperature
                             .TEMPERATURE_MODEL_PARAMETERS['sapm'][
                                 'close_mount_glass_glass'])
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global,
                                               temperature, windspeed,
                                               **temp_model_params)
    elif type_pv == 'open_rack':
        temp_model_params = (pvlib.temperature.
                             TEMPERATURE_MODEL_PARAMETERS['sapm'][
                                 'open_rack_glass_polymer'])
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global,
                                               temperature, windspeed,
                                               **temp_model_params)
    else:
        raise ValueError('Invalid type of PV panel installation')

    effective_irradiance = pvsystem.sapm_effective_irradiance(
                                irradiance.poa_direct, irradiance.poa_diffuse,
                                airmass_absolute, aoi, multi_si)

    # Multi-Si Sandia PV Array Performance Model (SAPM)
    dc = pvsystem.sapm(effective_irradiance, celltemp, multi_si)

    # Multi-Si specifications
    Imp_multiSi = multi_si.Impo  # 8.1243
    Vmp_multiSi = multi_si.Vmpo  # 29.1988
    kWp = (Imp_multiSi * Vmp_multiSi) / 1000

    # Use PV Watts model to convert DC to AC power (inverter model)
    sapm_inverters = pvsystem.retrieve_sam('cecinverter')
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

    # Convert to kW and fill NaN values
    ac_mod = ac.fillna(0) / 1000
    ac_mod[ac_mod < 0] = 0

    # Generate profile per 1 kWp
    total_profile_tmy = (1/kWp) * ac_mod

    print('Calculated PV profile with a capacity factor of '
          f'{round(total_profile_tmy.mean(), 4)}')

    return total_profile_tmy



