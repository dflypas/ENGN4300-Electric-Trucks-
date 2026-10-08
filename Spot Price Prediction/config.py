from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

PRICE_DEMAND_GLOB = str(DATA_DIR / "PRICE_AND_DEMAND_*_NSW1.csv")
FUELMIX_MONTHLY_FILE = DATA_DIR / "nsw_monthly_fuelmix.csv"
ISP_DEMAND_FILE = DATA_DIR / "isp_demand_forecast.csv"
GENERATOR_FILE = DATA_DIR / "generator_info.xlsx"

REGION = "NSW1"

# NEM market price floor is fixed by rule
MARKET_PRICE_FLOOR = -1000.0

# Market Price Cap (MPC) is on a legislated staged increase (AEMC Dec 2023
# rule change) through FY2027-28, then reverts to ordinary CPI indexation.
# Financial year (July-June) -> MPC, $/MWh. Confirmed values through FY26-27;
# FY27-28 completes the staged increase (assumed here as a further step
# consistent with the pathway - confirm against AEMC's FY27-28 determination
# when published) and years beyond are approximated at 3%/yr CPI indexation.
MPC_BY_FINANCIAL_YEAR = {
    2025: 17500.0,   # FY2024-25
    2026: 20300.0,   # FY2025-26
    2027: 23200.0,   # FY2026-27
    2028: 25000.0,   # FY2027-28 - APPROXIMATE, final staged step not yet confirmed
}
_CPI_ASSUMPTION = 1.03


def market_price_cap(calendar_year: int) -> float:
    """MPC applicable for a given calendar year (approximated as that
    year's mid-financial-year cap, i.e. Jan-Jun uses the FY ending that
    year, Jul-Dec uses the FY starting that year - simplified here to just
    use the FY-ending value for the whole calendar year, which is accurate
    to within one cap step)."""
    fy_end = calendar_year  # FY2025-26 "ends" mid-2026; treat calendar_year as FY-end label
    if fy_end in MPC_BY_FINANCIAL_YEAR:
        return MPC_BY_FINANCIAL_YEAR[fy_end]
    last_known_fy = max(MPC_BY_FINANCIAL_YEAR)
    years_beyond = fy_end - last_known_fy
    if years_beyond < 0:
        return MPC_BY_FINANCIAL_YEAR[min(MPC_BY_FINANCIAL_YEAR)]
    return MPC_BY_FINANCIAL_YEAR[last_known_fy] * (_CPI_ASSUMPTION ** years_beyond)

# Quantiles the model predicts
QUANTILES = [0.1, 0.5, 0.9]

# Fuel-mix columns (GWh) from the OpenElectricity monthly export, and the
# technology bucket we roll each one into for feature purposes.
FUELMIX_TECH_MAP = {
    "Coal (Black) -  GWh": "coal",
    "Gas (CCGT) -  GWh": "gas",
    "Gas (OCGT) -  GWh": "gas",
    "Distillate -  GWh": "gas",  # peaking liquid fuel, lumped with gas peakers
    "Bioenergy (Biomass) -  GWh": "other",
    "Hydro -  GWh": "hydro",
    "Wind -  GWh": "wind",
    "Solar (Utility) -  GWh": "solar_utility",
    "Solar (Rooftop) -  GWh": "solar_rooftop",
    "Battery (Discharging) -  GWh": "battery",
}
