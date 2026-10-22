import pandas as pd

# ==============================
# SETTING UP FILE READING
# ==============================

PARAMETER_FILE = r"C:\Users\joels\Desktop\UNI Folder\Year 4 Semester 2\Capstone\data_import_template.csv"

# Load the table
parameters = pd.read_csv(PARAMETER_FILE)

#Force convert the value column to numbers
parameters["value"] = pd.to_numeric(parameters["value"], errors="coerce")

# Create a dictionary
param = parameters.set_index("field_name")["value"].to_dict()


# ==============================
# RETRIEVING PARAMETER VALUES
# ==============================

required_fields = [
    "grid_electricity_price",
    "feed_in_tariff",
    "fleet_size",
    "ev_retrofit_cost",
    "diesel_fuel_consumption_rate",
    "vehicle_residual_value",
    "ev_maintenance_cost",
    "diesel_maintenance_cost",
    "discount_rate",
    "inflation_rate",
    "study_period_T",
    "num_ev_trucks",
    "num_diesel_trucks",
    "num_pv_panels",
    "num_chargers",
    "num_swappable_batteries",
    "grid_import_t",
    "grid_export_t"
]

for field in required_fields:
    globals()[field] = param.get(field, 0)

# ==================================
# SETTING UP CONSTANT VALUES
# ==================================

price_battery = 0 #Price per battery

price_per_panel = 0 #Price per panel $/kW

price_charger = 0 #Price per charger

diesel_price = 0 #diesel price $/L

energy_per_panel = 0 #Watts per panel W

price_forklift = 0 #price of a forklift


# ==============
#  INITIAL COST
# ==============

def calculate_initial_cost():
    return (
        num_pv_panels * price_per_panel * energy_per_panel #number of panels to be optimised
        + num_swappable_batteries * price_battery #batteries  to be optimised
        + num_chargers * price_charger #chargers to be optimised
        + num_ev_trucks * ev_retrofit_cost  #number of ev trucks to be optimised
        + price_forklift
    )


# ==============
#  ONGOING COST
# ==============

def calculate_ongoing_cost():
    return (
        grid_import_t * grid_electricity_price #grid importing to be optimised
        - grid_export_t * feed_in_tariff #grid exporting to be optimised
        + diesel_fuel_consumption_rate * num_diesel_trucks * diesel_price #number of diesel trucks to be optimised
        + num_diesel_trucks * diesel_maintenance_cost #number of diesel trucks to be optimised
        + num_ev_trucks * ev_maintenance_cost #number of EVs to be optimised
    )

# ==============
#  FINAL PRINT
# ==============


initial_cost = calculate_initial_cost()
ongoing_cost = calculate_ongoing_cost()


print(f"Initial cost: ${initial_cost:,.2f}")
print(f"Ongoing cost: ${ongoing_cost:,.2f}")
