#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:15:32 2026

@author: mollyrobertson
"""

    
    
    # ============================================================
    # Objective Function Definition
    # ============================================================
    start_time = time.time()
    
    # OBJECTIVE 1: COSTS
    # ============================================================
    # OPERATION
    # ============================================================
    
    # Annual cost of electricity imported from grid
    an_op_grid_imp = gp.quicksum(P_G_imp[t] * dt * spot_price_t[t] / 1000 for t in times)
    
    # Annual revenue from electricity exported to grid
    an_op_grid_exp = gp.quicksum(P_G_exp[t] * dt * sell_price_t[t] / 1000 for t in times)
    
    # Diesel truck energy cost
    an_op_diesel = gp.quicksum((route_return[r] - route_departure[r]) * diesel_truck_consumption * diesel_cost_per_kwh * (1 - z[route_truck[r], route_year[r]]) for r in routes)
    
    # Total annual operating cost
    an_op = (an_op_grid_imp - an_op_grid_exp + an_op_diesel)
    
    
    # ============================================================
    # CAPITAL AND INVESTMENTS
    # ============================================================
    
    # Capital recovery factor
    crf = (discount_rate * (1 + discount_rate) ** project_lifetime / ((1 + discount_rate) ** project_lifetime - 1))
    
    # Swappable batteries
    an_capex_SB = (crf * battery_cost * N_SB)
    
    # Chargers
    an_capex_CS = (crf * charger_cost * N_CS)
    
    # Stationary BESS
    an_capex_BESS = (crf * bess_capex_per_kwh * Q_BESS)
    
    # PV
    an_capex_PV = (crf * pv_capex_per_kw * E_PV_cap)
    
    # Truck retrofits
    an_capex_retrofit = (crf * retrofit_cost * gp.quicksum(k[i, y] for i in trucks for y in years))
    
    # Total annualised CAPEX
    an_capex = (an_capex_SB + an_capex_CS + an_capex_BESS + an_capex_PV + an_capex_retrofit)
    
    
    # ============================================================
    # REPLACEMENTS
    # ============================================================
    
    # Swappable batteries
    an_rep_SB = (replacement_annualisation(battery_cost, project_lifetime, discount_rate, sb_lifetime) * N_SB)
    
    # Chargers
    an_rep_CS = (replacement_annualisation(charger_cost, project_lifetime, discount_rate, charger_lifetime) * N_CS)
    
    # Stationary BESS
    an_rep_BESS = (replacement_annualisation(bess_capex_per_kwh, project_lifetime, discount_rate, bess_lifetime) * Q_BESS)
    
    # PV
    an_rep_PV = (replacement_annualisation(pv_capex_per_kw, project_lifetime, discount_rate, pv_lifetime) * E_PV_cap)
    
    # Truck retrofits
    an_rep_retrofit = (replacement_annualisation(retrofit_cost, project_lifetime, discount_rate, retrofit_lifetime) * gp.quicksum(k[i, y] for i in trucks for y in years))
    
    # Total annualised replacement costs
    an_rep = (an_rep_SB + an_rep_CS + an_rep_BESS + an_rep_PV + an_rep_retrofit)
    
    
    # ============================================================
    # FIXED O&M
    # ============================================================
    
    # Swappable batteries
    an_om_SB = (battery_cost * N_SB * sb_om)
    
    # Chargers
    an_om_CS = (charger_cost * N_CS * charger_om)
    
    # Stationary BESS
    an_om_BESS = (bess_capex_per_kwh * Q_BESS * bess_om)
    
    # PV
    an_om_PV = (pv_capex_per_kw * E_PV_cap * pv_om)
    
    # Electric truck retrofit O&M
    an_om_retrofit = (retrofit_cost * retrofit_om * gp.quicksum(z[i, y] for i in trucks for y in years))
    
    # Diesel O&M
    an_om_DIESEL = gp.quicksum(diesel_capex * diesel_om * (1 - z[i, y]) for y in years for i in trucks_in_year[y])
    
    # Total annual O&M cost
    an_om = (an_om_SB + an_om_CS + an_om_BESS + an_om_PV + an_om_retrofit + an_om_DIESEL)
    
    
    # ============================================================
    # TOTAL COST OBJECTIVE
    # ============================================================
    
    obj1 = (an_op + an_capex + an_rep + an_om)
    m.setObjective(obj1, GRB.MINIMIZE)
    
    
    # ============================================================
    # Optimise
    # ============================================================
    
    start_time = time.time()
    
    m.optimize()
    
    end_time = time.time()
    print(f"Optimisation: {end_time - start_time:.2f} seconds")
        
        


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
        discount_rate * (1 + discount_rate) ** project_lifetime
        / ((1 + discount_rate) ** project_lifetime - 1)
    )

    return replacement_present_value * crf



