#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:29:47 2026

@author: mollyrobertson
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib import pvsystem
import datetime as dt



class EnergyDataProcessor:
    """
    Class to handle data processing for energy optimization.

    Attributes:
        assess_country (str): The country being assessed.
        get_year (int): The year of assessment.
        lat (float): self.lat of the location.
        lon (float): self.lon of the location.
        fit (float): Feed-in-tariff.
        constrained (boolean): if True, the MES is constrained by limitations in terms of capacities that can be installed.
        residential sector (boolean): if True, the residential sector is included.
    """
    
    def __init__(self, assess_country, get_year, lat, lon, fit, constrained, residential_sector):
        """
        Initializes the EnergyDataProcessor class.

        Parameters:
            assess_country (str): The country being assessed.
            get_year (int): The year of assessment.
            lat (float): self.lat of the location.
            lon (float): self.lon of the location.
            fit (float): Feed-in-tariff.
            constrained (boolean): if True, the MES is constrained by limitations in terms of capacities that can be installed.
            residential sector (boolean): if True, the residential sector is included.
        """
        self.assess_country = assess_country
        self.get_year = get_year
        self.lat = lat
        self.lon = lon
        self.fit = fit
        self.constrained = constrained
        self.residential_sector = residential_sector


    def get_weather_data(self, start_year=2007, end_year=2016):
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
        data['time(UTC)'] = data['time(UTC)'].apply(lambda x: str(self.get_year) + x[4:])
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
                             module_spec="Sharp_NDQ235F4__2013_", type_pv = "open_rack"):
    """
    Calculates PV generation per kWp.

    Args:
        weather_data (dataframe): dataframe with weather information for one year [pd.Dataframe].
        latitude (float): latitude, in decimal degrees, between -90 and 90, north is positive (ISO 19115) [degrees].
        longitude (float): longitude, in decimal degrees, between -180 and 180, east is positive (ISO 19115) [degrees].
        module_spec (str): type of solar module to be modelled [str] default is Sharp_NDQ235F4__2013_. The PV module modelled, here standard the Sharp, since parameters are available and multi-Si
        type_pv (str): str, default is roof_mounted. How the module will be installed, here assumed to be roof mounted. However, can be modified in the future
    Returns:
        total_profile_tmy (dataframe): the PV profile per kWp panel modelled, for a TMY.
    """
    
    # Assuming that panels are oriented to the south with angle of 35 deg to surface
    sa = 180 #Surface azimuth angle, 180 (south), 270(west), 90(east)
    st = 35 #angle the roof makes with the surface, assumed to be 35 degrees
    
    # If TMY, delete first substrings from rows to avoid inconsitencies
    temperature = weather_data['temp_air'].astype(float)
    windspeed = weather_data['wind_speed'].astype(float)
    ghi = weather_data['ghi']
    dni = weather_data['dni']
    dhi = weather_data['dhi']
    pressure = weather_data['pressure'].astype(float)
    
    #solar details for location
    sp = pvlib.solarposition.get_solarposition(weather_data.index, float(latitude), float(longitude), pressure=pressure)
    irradiance = pvlib.irradiance.get_total_irradiance(st, sa, sp.apparent_zenith, sp.azimuth, dni, ghi, dhi)
    
    # Get db with modules, choose the one in the function and get specs
    modules = pvsystem.retrieve_sam('SandiaMod').T
    modules_multi_si = modules[modules['Material']=="mc-Si"]
    multi_si = modules_multi_si[modules_multi_si.index.str.contains(module_spec)].T
    multi_si = multi_si[module_spec]
    
    #relative airmass and aoi
    airmass = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith)
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass, pressure=pressure)

    #aoi and celltemp
    aoi = pvlib.irradiance.aoi(st, sa, sp.zenith, sp.azimuth)

    if type_pv == "roof_mounted":
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global, temperature,windspeed, **temperature_model_parameters)
    elif type_pv == "open_rack":
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
        celltemp = pvlib.temperature.sapm_cell(irradiance.poa_global, temperature,windspeed, **temperature_model_parameters)
    else:
        raise ValueError("Invalid type of PV panel installation")
        
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(irradiance.poa_direct, irradiance.poa_diffuse, 
                                                                    airmass_absolute, aoi, multi_si)
    
    #Multi-Si
    dc = pvlib.pvsystem.sapm(effective_irradiance, celltemp, multi_si)

    #Multi-Si specifications
    Imp_multiSi = multi_si.Impo #8.1243
    Vmp_multiSi = multi_si.Vmpo #29.1988
    kWp = (Imp_multiSi * Vmp_multiSi) / 1000
    
    #use pv Watts model to convert dc to ac power (inverter model)
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

    #convert to kW and fill NaN values
    ac_mod = ac.fillna(0) / 1000
    ac_mod[ac_mod < 0] = 0

    # Generate profile per 1 kWp
    total_profile_tmy = (1/kWp) * ac_mod
    
    print("Calculated PV profile with a capacity factor of '{}'".format(round(total_profile_tmy.mean(),4)))
    
    return total_profile_tmy






# TESTING

# Read input file
inputs = pd.read_csv("inputs.csv")
site = inputs.iloc[0]

processor = EnergyDataProcessor(
    assess_country=site["assess_country"],
    get_year=int(site["get_year"]),
    lat=float(site["lat"]),
    lon=float(site["lon"]),
    fit=float(site["fit"]),
    constrained=bool(site["constrained"]),
    residential_sector=bool(site["residential_sector"])
)

weather_data, elevation = processor.get_weather_data()
weather_data.to_csv("weather_output.csv")

# kW per kWp installed
pv_profile = processor.get_renewable_profiles(weather_data)
pd.DataFrame({"Time": weather_data.index,"PV_Profile": pv_profile}).to_csv("pv_profile_output.csv", index=False)







