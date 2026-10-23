import os
import pandas as pd
import streamlit as st
import altair as alt
from main import run_pipeline
import io
from contextlib import redirect_stdout
import re
import tempfile
import math


# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="DairyDrive Electric Optimiser", layout="wide")

# ── Default values come from input_data.csv ─────────────────
# This keeps the website defaults in sync with the model's own inputs.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
input_data = pd.read_csv(os.path.join(APP_DIR, "input_data.csv"))
defaults = input_data.set_index("field_name")["value"].to_dict()

# ── Theme styling ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Nunito+Sans:wght@400;600&display=swap');


/* Hide the anchor link icons next to headings */
[data-testid="stHeaderActionElements"],
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

html, body, [class*="css"]  {
    font-family: 'Nunito Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #2F4A43;
}
div.stButton > button {
    background-color: #2F4A43;
    color: white;
    border-radius: 25px;
    border: none;
    padding: 0.6em 2em;
    font-weight: 600;
    font-size: 1.05em;
}
div.stButton > button:hover {
    background-color: #3F5F57;
    color: white;
}
div[data-testid="stMetric"] {
    background-color: #C3D0CA;
    border-radius: 12px;
    padding: 15px;
}
.input-card-icon {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.input-card-icon svg {
    flex-shrink: 0;
}
.input-card-label {
    font-weight: 600;
    color: #2F4A43;
    font-size: 1.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:left; padding:10px 0 20px 0;">
<h1 style="font-size:3em; margin-bottom:0;">DairyDrive Electric</h1>
<p style="font-size:1.1em; color:#2F4A43;">Fleet Optimisation Demo</p>
</div>
""", unsafe_allow_html=True)

# ── Hero image ───────────────────────────────────────────────
hero_image_path = os.path.join(APP_DIR, "truck.jpg")
if os.path.exists(hero_image_path):
    st.image(hero_image_path, use_container_width=True)
else:
    st.info("Hero image not found. Add truck.jpg to the project folder to display it here.")

# ── Icon set (simple custom line icons, dark teal) ──────────
ICON_COLOR = "#2F4A43"


def make_icon(inner):
    """Wraps SVG shapes in the standard 28px teal line icon."""
    return (
        f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
        f'stroke="{ICON_COLOR}" stroke-width="2">{inner}</svg>'
    )


ICONS = {
    "duration": make_icon('<rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="16" y1="3" x2="16" y2="7"/>'),
    "truck": make_icon('<rect x="1" y="7" width="13" height="9" rx="1"/><polyline points="14,10 19,10 22,13 22,16 14,16"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/>'),
    "grid": make_icon('<polygon points="13,2 4,14 11,14 9,22 20,9 13,9"/>'),
    "price": make_icon('<circle cx="12" cy="12" r="9"/><line x1="12" y1="6" x2="12" y2="18"/><path d="M15 9 C15 7 9 7 9 10 C9 13 15 11 15 14 C15 17 9 17 9 15"/>'),
    "type": make_icon('<line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2" fill="#2F4A43"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2" fill="#2F4A43"/>'),
    "feedin": make_icon('<line x1="12" y1="20" x2="12" y2="5"/><polyline points="6,11 12,5 18,11"/><line x1="4" y1="22" x2="20" y2="22"/>'),
    "pv": make_icon('<circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/>'),
}


ICONS["location"] = make_icon('<path d="M12 22 C12 22 5 14 5 9 A7 7 0 0 1 19 9 C19 14 12 22 12 22 Z"/><circle cx="12" cy="9" r="2.5"/>')


def input_card(icon_key, label, widget_fn):
    """Renders an icon + label header inside a bordered card, then the
    input widget itself."""
    with st.container(border=True):
        st.markdown(
            f'<div class="input-card-icon">{ICONS[icon_key]}<span class="input-card-label">{label}</span></div>',
            unsafe_allow_html=True,
        )
        return widget_fn()


# ── Inputs ───────────────────────────────────────────────────
# Widgets use label_visibility="collapsed" because the card header is the label.
st.subheader("Inputs")

# ---- Study set-up ----
st.markdown("##### Study set-up")
setup_cols = st.columns(2)

with setup_cols[0]:
    duration_days = input_card(
        "duration", "Duration (days)",
        lambda: st.number_input(
            "Duration", min_value=1, max_value=730,
            value=int(defaults["duration"]), step=1,
            key="duration_input", label_visibility="collapsed",
            help="Days simulated (max 730). Longer runs solve much more slowly.",
        ),
    )

with setup_cols[1]:
    pv_max_kw = input_card(
        "pv", "PV space (kW)",
        lambda: st.number_input(
            "PV space", min_value=0.0,
            value=float(defaults["PV_max_kw"]), step=250.0,
            key="pv_input", label_visibility="collapsed",
            help="Maximum solar capacity that can be installed, in kW.",
        ),
    )
# Later: swap this for a roof area (m2) input and convert it to kW here.

# ---- Location ----
# Used to build the solar (PV) profile from weather data for this site.
st.markdown("##### Location")
location_cols = st.columns(2)

with location_cols[0]:
    latitude = input_card(
        "location", "Latitude (degrees)",
        lambda: st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=float(defaults["latitude"]), step=0.001, format="%.3f",
            key="latitude_input", label_visibility="collapsed",
            help="Site latitude. Southern hemisphere is negative. Example: -36.674",
        ),
    )

with location_cols[1]:
    longitude = input_card(
        "location", "Longitude (degrees)",
        lambda: st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=float(defaults["longitude"]), step=0.001, format="%.3f",
            key="longitude_input", label_visibility="collapsed",
            help="Site longitude. Example: 149.842",
        ),
    )

# ---- Grid connection ----
st.markdown("##### Grid connection")
grid_cols = st.columns(4)

with grid_cols[0]:
    grid_max_import = input_card(
        "grid", "Max grid import (kW)",
        lambda: st.number_input(
            "Max grid import", min_value=0.0,
            value=float(defaults["GRID_max_import"]), step=100.0,
            key="grid_import_input", label_visibility="collapsed",
        ),
    )

with grid_cols[1]:
    grid_max_export = input_card(
        "grid", "Max grid export (kW)",
        lambda: st.number_input(
            "Max grid export", min_value=0.0,
            value=float(defaults["GRID_max_export"]), step=100.0,
            key="grid_export_input", label_visibility="collapsed",
        ),
    )

with grid_cols[2]:
    grid_type_label = input_card(
        "type", "Grid connection type",
        lambda: st.selectbox(
            "Grid connection type",
            ["Regular grid connection", "Wholesale market (spot price)"],
            index=int(defaults["Wholesale Market"]),
            key="grid_type_input", label_visibility="collapsed",
            help="Regular uses the fixed on-peak and off-peak prices below. "
                 "Wholesale market uses the spot price.",
        ),
    )

with grid_cols[3]:
    grid_upgrade_price = input_card(
        "price",
        "Grid upgrade cost ($)",
        lambda: st.number_input(
            "Grid upgrade cost",
            min_value=0.0,
            value=float(defaults["GRID_upgrade_price"]),
            step=1000.0,
            key="grid_upgrade_price_input",
            label_visibility="collapsed",
            help="Cost of upgrading the grid connection, in dollars.",
        ),
    )

# The model's "Wholesale Market" field is 0 for regular, 1 for spot price.
if grid_type_label == "Wholesale market (spot price)":
    wholesale_market = 1
else:
    wholesale_market = 0

# ---- Electricity prices (power bill) ----
st.markdown("##### Electricity prices (power bill)")
if wholesale_market == 1:
    st.caption("Wholesale market selected: the fixed on-peak and off-peak prices may not be used.")

price_cols = st.columns(2)

with price_cols[0]:
    onpeak_price = input_card(
        "price", "On-peak price ($/kWh)",
        lambda: st.number_input(
            "On-peak price", min_value=0.0,
            value=float(defaults["GRID_onpeak_price"]), step=0.01, format="%.3f",
            key="onpeak_input", label_visibility="collapsed",
        ),
    )

with price_cols[1]:
    offpeak_price = input_card(
        "price", "Off-peak price ($/kWh)",
        lambda: st.number_input(
            "Off-peak price", min_value=0.0,
            value=float(defaults["GRID_offpeak_price"]), step=0.01, format="%.3f",
            key="offpeak_input", label_visibility="collapsed",
        ),
    )

# ---- Feed-in tariff ----
# Time windows come from the descriptions in input_data.csv.
st.markdown("##### Feed-in tariff (selling power back)")
fit_cols = st.columns(3)

with fit_cols[0]:
    fit_on_peak = input_card(
        "feedin", "FiT 10am to 3pm ($/kWh)",
        lambda: st.number_input(
            "FiT 10am to 3pm", min_value=0.0,
            value=float(defaults["F_I_T_on_peak"]), step=0.01, format="%.3f",
            key="fit_on_peak_input", label_visibility="collapsed",
        ),
    )

with fit_cols[1]:
    fit_off_peak = input_card(
        "feedin", "FiT 4pm to 9pm ($/kWh)",
        lambda: st.number_input(
            "FiT 4pm to 9pm", min_value=0.0,
            value=float(defaults["F_I_T_off_peak"]), step=0.01, format="%.3f",
            key="fit_off_peak_input", label_visibility="collapsed",
        ),
    )

with fit_cols[2]:
    fit_flat = input_card(
        "feedin", "FiT flat, 9pm to 10am ($/kWh)",
        lambda: st.number_input(
            "FiT flat", min_value=0.0,
            value=float(defaults["F_I_T_flat"]), step=0.01, format="%.3f",
            key="fit_flat_input", label_visibility="collapsed",
        ),
    )

# ---- Route schedule ----
st.markdown("##### Route schedule")
st.caption(
    "One row per route. Trucks are defined by the Truck No column: use the same "
    "number on several rows to give one truck multiple routes in a day. "
    "Times are 24-hour (e.g. 8:00, 13:25)."
)
st.caption("To remove a route, tick the box at the left of the row and press Delete. Use the + at the bottom to add one.")

EDIT_COLUMNS = ["Name", "Vol", "Pick Up Time", "Drop Off Time", "Truck No", "Diesel Used"]

SAMPLE_SCHEDULE = pd.DataFrame(
    [
        ["Bob", 1000, "8:00", "10:35", 1, 50],
        ["Bill", 400, "8:00", "9:05", 2, 23],
        ["Ben", 2000, "8:00", "12:00", 3, 80],
        ["Brad", 1700, "8:00", "11:25", 4, 60],
        ["Belinda", 2000, "8:00", "12:25", 5, 95],
        ["Barry", 2500, "8:00", "13:25", 6, 100],
        ["Betty", 300, "13:00", "18:00", 7, 83],
        ["Bruce", 2000, "14:00", "17:05", 2, 54],
        ["Bart", 1000, "15:00", "17:55", 1, 45],
        ["Becka", 1500, "16:00", "19:10", 8, 60],
        ["Bella", 1600, "16:00", "18:45", 4, 35],
        ["Brianna", 2000, "16:00", "17:50", 3, 25],
    ],
    columns=EDIT_COLUMNS,
)


def load_default_schedule():
    """Start from route_schedule_2.csv if it has the expected columns."""
    path = os.path.join(APP_DIR, "route_schedule_2.csv")
    try:
        df = pd.read_csv(path)
        if all(c in df.columns for c in EDIT_COLUMNS):
            return df[EDIT_COLUMNS].copy()
    except Exception:
        pass
    return SAMPLE_SCHEDULE.copy()


if "schedule_df" not in st.session_state:
    st.session_state["schedule_df"] = load_default_schedule()

schedule_df = st.data_editor(
    st.session_state["schedule_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="schedule_editor",
    column_config={
        "Name": st.column_config.TextColumn("Name"),
        "Vol": st.column_config.NumberColumn("Volume (L)", min_value=0, step=100),
        "Pick Up Time": st.column_config.TextColumn("Pick up (HH:MM)"),
        "Drop Off Time": st.column_config.TextColumn("Drop off (HH:MM)"),
        "Truck No": st.column_config.NumberColumn("Truck No", min_value=1, step=1),
        "Diesel Used": st.column_config.NumberColumn("Diesel used (L)", min_value=0),
    },
)

TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
REQUIRED = ["Pick Up Time", "Drop Off Time", "Truck No", "Diesel Used"]


def to_hours(s):
    h, m = str(s).strip().split(":")
    return int(h) + int(m) / 60


def validate_schedule(df):
    """Returns a list of human-readable problems (empty list = valid)."""
    errors = []
    if df.empty:
        return ["Add at least one route."]
    for idx, row in df.iterrows():
        n = idx + 1
        if row[REQUIRED].isna().any():
            errors.append(f"Row {n}: fill in truck, pick up, drop off and diesel used.")
            continue
        pu, do = str(row["Pick Up Time"]).strip(), str(row["Drop Off Time"]).strip()
        if not (TIME_RE.match(pu) and TIME_RE.match(do)):
            errors.append(f"Row {n}: times must look like 8:00 or 13:25.")
        elif math.ceil(to_hours(do)) <= math.floor(to_hours(pu)):
            errors.append(f"Row {n}: drop off must be after pick up.")
    if not errors:
        for truck, grp in df.groupby("Truck No"):
            spans = sorted(
                (math.floor(to_hours(a)), math.ceil(to_hours(b)))
                for a, b in zip(grp["Pick Up Time"], grp["Drop Off Time"])
            )
            for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                if s2 < e1:
                    errors.append(
                        f"Truck {int(truck)} has overlapping routes (the model works in whole hours, "
                        f"so allow a full hour between one drop off and the next pick up)."
                    )
                    break
    return errors


def is_blank(s):
    return s.isna() | (s.astype(str).str.strip() == "")


# Ignore completely empty rows (e.g. a freshly added row)
blank_rows = (is_blank(schedule_df["Pick Up Time"]) & is_blank(schedule_df["Drop Off Time"])
              & is_blank(schedule_df["Truck No"]) & is_blank(schedule_df["Diesel Used"]))
clean_df = schedule_df[~blank_rows].reset_index(drop=True)

schedule_errors = validate_schedule(clean_df)
for e in schedule_errors:
    st.error(e)

if not schedule_errors:
    n_trucks_from_schedule = int(clean_df["Truck No"].nunique())
    st.caption(f"{len(clean_df)} routes across {n_trucks_from_schedule} trucks.")

# ── Run controls ──────────────────────────────────────────────
st.write("")
demo_mode = st.checkbox("Quick preview (faster, less accurate)", value=True)

if demo_mode:
    time_limit = 200
    mip_gap = 0.01
else:
    time_limit = None
    mip_gap = 0.01

# Keys match the field_name column in input_data.csv.
overrides = {
    "duration": duration_days,
    "PV_max_kw": pv_max_kw,
    "latitude": latitude,
    "longitude": longitude,
    "GRID_max_import": grid_max_import,
    "GRID_max_export": grid_max_export,
    "GRID_upgrade_price": grid_upgrade_price,
    "Wholesale Market": wholesale_market,
    "GRID_onpeak_price": onpeak_price,
    "GRID_offpeak_price": offpeak_price,
    "F_I_T_on_peak": fit_on_peak,
    "F_I_T_off_peak": fit_off_peak,
    "F_I_T_flat": fit_flat,
}

if not schedule_errors:
    overrides["n_trucks"] = n_trucks_from_schedule

run_button = st.button("Run Optimisation", disabled=bool(schedule_errors))

# ── Run pipeline ────────────────────────────────────────────
if run_button:
    # Write only the columns the route processor reads
    full = clean_df.copy()
    full["Truck No"] = full["Truck No"].astype(int)
    full["Pick Up Time"] = full["Pick Up Time"].astype(str).str.strip()
    full["Drop Off Time"] = full["Drop Off Time"].astype(str).str.strip()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    full.to_csv(tmp.name, index=False)

    try:
        with st.spinner("Solving... this may take a moment"):
            results, df_results = run_pipeline(
                overrides, time_limit=time_limit, mip_gap=mip_gap, route_csv=tmp.name
            )
    finally:
        os.remove(tmp.name)

    if results is None:
        st.session_state.pop("results", None)
        st.session_state.pop("df_results", None)
        st.error("No feasible solution found. Try widening the input ranges.")
    else:
        st.success(f"Optimisation complete: {results['status']}")
        st.session_state["results"] = results
        st.session_state["df_results"] = df_results

# ── Results display ─────────────────────────────────────────
def money(value):
    """Formats a dollar amount in plain language, e.g. $1.25M or $48,000."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


def group_costs(items):
    """Turns the model's five NPV cost lines into four easy-to-read groups."""
    grouped = {}
    grouped["Upfront capital"] = items["Upfront CAPEX"]
    grouped["Running costs (energy and diesel)"] = items["Operating costs (PV)"]
    grouped["Maintenance"] = items["Fixed O&M (PV)"]
    grouped["Replacements (net of resale value)"] = (
        items["Replacements (PV)"] + items["Residual value (PV)"]
    )
    return grouped


if "results" in st.session_state:
    results = st.session_state["results"]

    st.markdown("""
    <div style="background-color:#E8EEEA; padding:25px; border-radius:8px; margin:20px 0;">
    <h2 style="text-align:center; margin:0;">Results</h2>
    </div>
    """, unsafe_allow_html=True)

    years = results["horizon_years"]
    bau_total = results["bau_npv_total"]
    opt_total = results["npv_total"]
    saving = bau_total - opt_total
    saving_pct = 100 * saving / bau_total

    # ---- Headline numbers ----
    st.subheader(f"Total cost over {years} year(s)")
    head1, head2, head3 = st.columns(3)
    head1.metric("Business as usual (all diesel)", money(bau_total))
    head2.metric("Optimised electric fleet", money(opt_total))
    if saving >= 0:
        head3.metric("Saving from electrifying", money(saving), f"{saving_pct:.1f}% cheaper")
    else:
        head3.metric("Extra cost of electrifying", money(-saving), f"{-saving_pct:.1f}% dearer", delta_color="inverse")
    st.caption("All costs are in today's dollars (net present value) at the chosen discount rate.")

    # ---- Bar chart: BAU vs optimised, split by cost type ----
    bau_groups = group_costs(results["bau_npv_items"])
    opt_groups = group_costs(results["npv_items"])

    scenario_names = ["Business as usual", "Optimised electric fleet"]
    chart_rows = []
    for component in bau_groups:
        chart_rows.append({"Scenario": scenario_names[0], "Cost type": component, "Cost": bau_groups[component]})
        chart_rows.append({"Scenario": scenario_names[1], "Cost type": component, "Cost": opt_groups[component]})
    chart_df = pd.DataFrame(chart_rows)

    component_order = list(bau_groups.keys())
    component_colours = ["#2F4A43", "#C8960C", "#7FA89B", "#C3D0CA"]

    cost_chart = (
        alt.Chart(chart_df)
        .mark_bar(size=110)
        .encode(
            x=alt.X("Scenario:N", sort=scenario_names, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
            y=alt.Y("Cost:Q", title="Cost over project life (AUD)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Cost type:N",
                sort=component_order,
                scale=alt.Scale(domain=component_order, range=component_colours),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            order=alt.Order("Cost type:N", sort="ascending"),
            tooltip=["Scenario", "Cost type", alt.Tooltip("Cost:Q", format="$,.0f")],
        )
        .properties(height=420)
    )
    st.altair_chart(cost_chart, use_container_width=True)

    # ---- What the optimiser chose ----
    st.subheader("What the optimiser chose")
    pick1, pick2, pick3 = st.columns(3)
    pick1.metric(
        "Trucks switched to electric",
        f"{results['trucks_electrified']:.0f} of {results['fleet_size']:.0f}",
    )
    pick2.metric("Swappable batteries", f"{results['number_of_batteries']:.0f}")
    pick3.metric("Chargers", f"{results['number_of_chargers']:.0f}")

    pick4, pick5, pick6 = st.columns(3)
    pick4.metric("Solar (PV) installed", f"{results['PV_capacity_kW']:,.0f} kW")
    pick5.metric("Stationary battery (BESS)", f"{results['BESS_capacity_kWh']:,.0f} kWh")
    pick6.metric("Grid connection", f"{results['grid_capacity_kW']:,.0f} kW")

    # ---- Detail for anyone who wants it ----
    with st.expander("Detailed numbers"):
        st.write(f"Solve time: {results['solve_time_seconds']:.1f} seconds")
        summary = results.get("summary_text")
        if summary:
            st.code(summary, language=None)
            st.download_button(
                "Download summary (.txt)",
                data=summary,
                file_name="optimisation_summary.txt",
                mime="text/plain",
            )