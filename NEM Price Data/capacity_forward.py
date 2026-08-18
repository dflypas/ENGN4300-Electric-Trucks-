import re
import pandas as pd
from config import GENERATOR_FILE

INCLUDE_STATUSES = ["In Service", "Committed", "Committed*", "In Commissioning",
                     "Anticipated", "Publicly Announced"]

TECH_MAP = {
    "Coal": "coal",
    "Gas Turbine": "gas",
    "Hydro": "hydro",
    "Wind": "wind",
    "Solar PV": "solar_utility",
    "Battery Storage": "battery",
    "Other": "other",
}


def load_generator_information() -> pd.DataFrame:
    df = pd.read_excel(GENERATOR_FILE, sheet_name="Generator Information", header=3)
    df.columns = df.columns.astype(str).str.strip()
    return df


def build_forward_capacity_by_year(region: str = "NSW1") -> pd.DataFrame:
    df = load_generator_information()
    region_short = region.replace("1", "")  # "NSW1" file values are like "NSW1" too, keep as-is
    nsw = df[df["Region"].astype(str).str.contains(region_short, na=False)].copy()
    nsw = nsw[nsw["Commitment Status"].isin(INCLUDE_STATUSES)]
    nsw = nsw[nsw["Technology Type"].isin(TECH_MAP.keys())]
    nsw["tech"] = nsw["Technology Type"].map(TECH_MAP)

    # Find every "Scheduled Capacity (MW) Summer Typical <year>-<year+1>" column
    summer_cols = [c for c in nsw.columns if re.match(r"Scheduled Capacity \(MW\) Summer Typical \d{4}-\d{2}", c)]

    rows = []
    for col in summer_cols:
        year = int(re.search(r"(\d{4})-\d{2}", col).group(1)) + 1  # label by the year the summer falls mostly within
        by_tech = nsw.groupby("tech")[col].sum()
        for tech, mw in by_tech.items():
            rows.append({"year": year, "tech": tech, "capacity_mw": mw})

    cap = pd.DataFrame(rows)
    cap_wide = cap.pivot(index="year", columns="tech", values="capacity_mw").fillna(0.0)
    cap_wide = cap_wide.sort_index()
    return cap_wide


def capacity_to_shares(cap_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert MW capacity by tech/year into the same *_share feature naming
    used by the historical fuel-mix module, so the forward feature table
    lines up column-for-column with training. Capacity share is a proxy for
    generation share - not exact (capacity factors differ by tech) but
    consistent with what the historical model was trained to relate to
    price via the residual-demand mechanism."""
    shares = cap_wide.div(cap_wide.sum(axis=1), axis=0)
    shares = shares.rename(columns={
        "coal": "coal_share", "gas": "gas_share", "hydro": "hydro_share",
        "wind": "wind_share", "solar_utility": "solar_utility_share",
        "battery": "battery_share", "other": "other_share",
    })
    return shares


if __name__ == "__main__":
    cap = build_forward_capacity_by_year()
    print("NSW1 scheduled capacity by technology, MW:")
    print(cap.round(0))
    print("\nAs generation-mix shares:")
    print(capacity_to_shares(cap).round(3))
