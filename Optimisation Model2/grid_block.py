"""
grid_block.py

Adds grid connection sizing and import/export mode constraints to the
shared Gurobi model. The grid_import and grid_export variables themselves
are created in the optimiser wrapper (as addMVar arrays), since other
blocks (BESS) also need to reference them. This block adds the
constraints that size and bound them: installed connection capacity,
network import/export limits, and mutual exclusivity between importing
and exporting in the same timestep.

Rewritten with addMVar so each constraint family is one array-level
addConstr call instead of one call per timestep - see the note at the
top of battery_storage_block.py for why this matters at large timestep
counts.

This block only adds variables and constraints. Solving is handled by the
optimiser wrapper, not here.
"""

import numpy as np
from gurobipy import GRB


def build_grid_block(model, times, dt, grid_import, grid_export,
    grid_existing_kw, grid_max_kw, grid_import_max, grid_export_max):

    n_t = len(times)

    # grid_import_max/grid_export_max arrive as {t: value} dicts from
    # main.py; convert once to plain arrays in time order.
    import_max_array = np.array([grid_import_max[t] for t in times])
    export_max_array = np.array([grid_export_max[t] for t in times])

    # Installed bidirectional grid connection capacity (kW)
    S_g = model.addVar(lb=grid_existing_kw, ub=grid_max_kw, vtype=GRB.CONTINUOUS, name="Grid_connection_kW")

    # 1 when import mode is active, 0 when export mode is active
    x_grid_import = model.addMVar(n_t, vtype=GRB.BINARY, name="Grid_import_mode")

    # Bounded by the installed connection capacity (scalar S_g broadcasts
    # against the length-n_t arrays - a normal scalar-vs-array broadcast)
    model.addConstr(grid_import <= S_g * dt, name="Grid_ConnectionImport")
    model.addConstr(grid_export <= S_g * dt, name="Grid_ConnectionExport")

    # Bounded by network limits, and mutually exclusive with export
    model.addConstr(grid_import <= import_max_array * dt * x_grid_import, name="Grid_ImportMode")
    model.addConstr(grid_export <= export_max_array * dt * (1 - x_grid_import), name="Grid_ExportMode")

    return {
        "S_g": S_g,
        "x_grid_import": x_grid_import,
    }