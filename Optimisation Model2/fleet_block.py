"""
DairyDrive Electric - Fleet conversion block

Adds truck conversion state variables and constraints to the shared Gurobi
model. Tracks which trucks are electric in each year, when each truck
converts, and the resulting count of electric vs diesel trucks per year.

This block only adds variables and constraints. Solving is handled by the
optimiser wrapper, not here.
"""

import gurobipy as gp
from gurobipy import GRB


def build_fleet_block(model, trucks, years, prev_y, first_y, initial_ev):

    fleet_size = len(trucks)

    # is_ev[i,y] = 1 if truck i has been converted to electric by year y
    is_ev = model.addVars(trucks, years, vtype=GRB.BINARY, name="is_ev")
    # converted[i,y] = 1 if truck i converts during year y
    converted = model.addVars(trucks, years, vtype=GRB.BINARY, name="converted")

    N_EV = model.addVars(years, lb=0, ub=fleet_size, vtype=GRB.INTEGER, name="N_EV")
    N_D = model.addVars(years, lb=0, ub=fleet_size, vtype=GRB.INTEGER, name="N_Diesel")

    for y in years:
        model.addConstr(N_EV[y] == gp.quicksum(is_ev[i, y] for i in trucks), name=f"Fleet_EVCount[{y}]")
        model.addConstr(N_D[y] + N_EV[y] == fleet_size, name=f"Fleet_Total[{y}]")

    for i in trucks:
        init = int(initial_ev.get(i, 0))

        # First modelled year: conversion is the change from the truck's known starting state
        if init == 0:
            model.addConstr(converted[i, first_y] == is_ev[i, first_y], name=f"ConvertFirst[{i}]")
        else:
            model.addConstr(is_ev[i, first_y] == 1, name=f"InitiallyEV[{i}]")
            model.addConstr(converted[i, first_y] == 0, name=f"NoFirstConversion[{i}]")

        # Once converted, a truck stays electric. converted[i,y] flags the year it happens.
        for y in years[1:]:
            py = prev_y[y]
            model.addConstr(is_ev[i, y] >= is_ev[i, py], name=f"ConversionMonotonic[{i},{y}]")
            model.addConstr(converted[i, y] == is_ev[i, y] - is_ev[i, py], name=f"ConversionEvent[{i},{y}]")

    return {
        "is_ev": is_ev,
        "converted": converted,
        "N_EV": N_EV,
        "N_Diesel": N_D,
    }