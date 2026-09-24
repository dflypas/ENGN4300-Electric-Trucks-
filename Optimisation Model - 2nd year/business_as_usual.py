#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:11:02 2026

@author: mollyrobertson
"""

"""
DairyDrive Electric
Business as Usual Model

Assumes:
    - All trucks remain diesel
    - All scheduled routes are completed by diesel trucks
    - No trucks are converted to electric
    - No swappable batteries
    - No chargers
    - No BESS
    - No PV
"""


import pandas as pd


def business_as_usual(
    routes,
    route_diesel_used,
    route_year,
    diesel_price_t,
    trucks,
    years,
    trucks_in_year,
    diesel_capex,
    diesel_lifetime,
    diesel_om,
    discount_rate,
    project_lifetime
):
    
    number_of_diesel_trucks = {y: len(trucks_in_year[y]) for y in years}

    diesel_used = sum(route_diesel_used[r] for r in routes)
    
    diesel_fuel_cost = sum(route_diesel_used[r] * diesel_price_t[route_year[r]] for r in routes)
    
    diesel_om_cost = sum(diesel_capex * diesel_om for y in years for i in trucks_in_year[y])
    
    
    crf = (discount_rate * (1 + discount_rate) ** project_lifetime / ((1 + discount_rate) ** project_lifetime - 1))

    diesel_capital_cost = (crf* diesel_capex* len(trucks))
    
    annualised_diesel_replacement = replacement_annualisation(diesel_capex, project_lifetime, discount_rate, diesel_lifetime)
    diesel_replacement_cost = sum(annualised_diesel_replacement * number_of_diesel_trucks[y] for y in years)
    
    total_cost = (diesel_fuel_cost + diesel_capital_cost + diesel_replacement_cost + diesel_om_cost)


    # ============================================================
    # Results
    # ============================================================
    
    results = {

        "total_cost": total_cost,

        # Cost breakdown
        "operation_cost": diesel_fuel_cost,
        "annualised_capex": diesel_capital_cost,
        "replacement_cost": diesel_replacement_cost,
        "OM_cost": diesel_om_cost,

        # Diesel
        "diesel_used_L": diesel_used,
        "diesel_fuel_cost": diesel_fuel_cost,

        # Fleet
        "number_of_electric_trucks": 0,
        "number_of_diesel_trucks": len(trucks),

        # Electric infrastructure
        "number_of_batteries": 0,
        "number_of_chargers": 0,
        "BESS_capacity_kWh": 0,
        "PV_capacity_kW": 0,
    }

    df_results = pd.DataFrame([results])

    df_results.to_csv(
        "business_as_usual_results.csv",
        index=False
    )

    return results, df_results



def replacement_annualisation(
    unit_cost,
    project_lifetime,
    discount_rate,
    technology_lifetime
):

    replacement_present_value = 0

    replacement_year = technology_lifetime

    while replacement_year < project_lifetime:

        replacement_present_value += (
            unit_cost
            / (1 + discount_rate) ** replacement_year
        )

        replacement_year += technology_lifetime

    crf = (
        discount_rate
        * (1 + discount_rate) ** project_lifetime
        / ((1 + discount_rate) ** project_lifetime - 1)
    )

    return replacement_present_value * crf



