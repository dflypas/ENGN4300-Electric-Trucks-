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

def replacement_annualisation(
    unit_cost,
    project_lifetime,
    discount_rate,
    technology_lifetime
):
    """
    Annualised replacement cost and annualised residual value, returned
    separately as (replacement, residual). Both are positive numbers.
    Net cost = replacement - residual.

    Assumes straight-line depreciation and constant replacement cost.
    """

    if project_lifetime <= 0:
        raise ValueError("Project lifetime must be positive.")

    if technology_lifetime <= 0:
        raise ValueError("Technology lifetime must be positive.")

    if discount_rate < 0:
        raise ValueError("Discount rate cannot be negative.")

    # Present value of replacements during the project.
    replacement_present_value = 0.0
    replacement_year = technology_lifetime

    while replacement_year < project_lifetime:
        replacement_present_value += (
            unit_cost / (1 + discount_rate) ** replacement_year
        )
        replacement_year += technology_lifetime

    # Remaining useful life of the asset at project end.
    asset_age_at_end = project_lifetime % technology_lifetime
    remaining_life = technology_lifetime - asset_age_at_end

    # No residual value if the asset has reached end of life.
    if asset_age_at_end == 0:
        residual_value = 0.0
    else:
        residual_value = (
            unit_cost
            * remaining_life
            / technology_lifetime
        )

    # Discount residual value back to the start of the project.
    residual_present_value = (
        residual_value / (1 + discount_rate) ** project_lifetime
    )

    # Capital recovery factor.
    if discount_rate == 0:
        crf = 1 / project_lifetime
    else:
        crf = (
            discount_rate * (1 + discount_rate) ** project_lifetime
            / ((1 + discount_rate) ** project_lifetime - 1)
        )

    return replacement_present_value * crf, residual_present_value * crf


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
    base_year = 0
    
    diesel_used = sum(route_diesel_used[r] for r in routes)

    if discount_rate == 0:
        crf = 1 / project_lifetime

    else:
        crf = (discount_rate * (1 + discount_rate) ** project_lifetime / ((1 + discount_rate) ** project_lifetime - 1))

    n_trucks = len(trucks)    
   # Per-truck annualised replacement and residual value
    rep_unit, res_unit = replacement_annualisation(
        diesel_capex, project_lifetime, discount_rate, diesel_lifetime
    )
 
    rows = []
    for y in years:
        fuel = diesel_used * diesel_price_t[y]
        capex = crf * diesel_capex * n_trucks
        rep = rep_unit * n_trucks
        res = res_unit * n_trucks
        om = diesel_capex * diesel_om * n_trucks
        total = fuel + capex + rep - res + om
        upfront_capex = diesel_capex * n_trucks       # face value, not annualised
        total_spend = upfront_capex + fuel + om 
 
        rows.append({
            "year_index": y,
            "year": base_year + y,
            "total_cost": total,
            "operation_cost": fuel,
            "annualised_capex": capex,
            "replacement_cost": rep,
            "residual_value": res,
            "OM_cost": om,
            "diesel_used_L": diesel_used,
            "diesel_price": diesel_price_t[y],
            "number_of_diesel_trucks": n_trucks,
            "number_of_electric_trucks": 0,
            "upfront_capex": upfront_capex,
            "total_spend": total_spend,
        })
 
    df = pd.DataFrame(rows)
    df["discount_factor"] = 1 / (1 + discount_rate) ** df["year_index"]
    df["discounted_cost"] = df["total_cost"] * df["discount_factor"]
    df["cumulative_cost"] = df["total_cost"].cumsum()
    return df
    