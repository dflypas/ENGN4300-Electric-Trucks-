#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:39:19 2026

@author: mollyrobertson
"""

    
    # ============================================================
    # Decision Variables
    # ============================================================
    
    N_EV = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_EV")
    N_D = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_D")
    N_SB = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(batteries), name="N_SB")
    N_CS = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(chargers), name="N_CS")
    
    z = m.addVars(trucks, years, vtype=GRB.BINARY, name="z")
    k = m.addVars(trucks, years, vtype=GRB.BINARY, name="k")
    w = m.addVars(batteries, vtype=GRB.BINARY, name="w")
    a = m.addVars(routes, batteries, vtype=GRB.BINARY, name="a")
    x_SB_ch = m.addVars(batteries, times, vtype=GRB.BINARY, name="x_SB_ch")
    
    E_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, name="E_SB")
    P_SB_ch = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, name="P_SB_ch")
    
    Q_BESS = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_BESS")
    P_BESS_ch_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_ch_inst")
    P_BESS_dis_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_dis_inst")
    E_BESS = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="E_BESS")
    P_BESS_ch = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_ch")
    P_BESS_dis = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_dis")
    x_BESS = m.addVars(times, vtype=GRB.BINARY, name="x_BESS")

    E_PV_cap = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=pv_max_kw, name="E_PV_cap")
    P_PV_generated = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_PV_generated")    
    
    S_G = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S_G")
    P_G_imp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_G_imp")
    P_G_exp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_G_exp")
    x_G = m.addVars(times, vtype=GRB.BINARY, name="x_G")
    
