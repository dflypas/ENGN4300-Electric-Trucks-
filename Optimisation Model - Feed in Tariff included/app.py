import streamlit as st
from main import run_pipeline

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="DairyDrive Electric Optimiser", layout="wide")

# ── Theme styling ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Nunito+Sans:wght@400;600&display=swap');

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
import os
hero_image_path = "truck.jpg"
if os.path.exists(hero_image_path):
    st.image(hero_image_path, use_container_width=True)
else:
    st.info("Hero image not found — add truck_hero.jpg to the project folder to display it here.")

# ── Icon set (simple custom line icons, dark teal) ──────────
ICON_COLOR = "#2F4A43"

ICONS = {
    "duration": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="16" y1="3" x2="16" y2="7"/></svg>',
    "battery": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="2"><rect x="2" y="7" width="17" height="10" rx="2"/><line x1="22" y1="10" x2="22" y2="14"/><rect x="5" y="10" width="4" height="4" fill="{ICON_COLOR}" stroke="none"/></svg>',
    "charger": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="2"><rect x="4" y="8" width="16" height="12" rx="2"/><line x1="9" y1="8" x2="9" y2="4"/><line x1="15" y1="8" x2="15" y2="4"/><polyline points="10,12 13,12 11,16 14,16"/></svg>',
    "pv": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="2"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/><line x1="4.5" y1="4.5" x2="6.5" y2="6.5"/><line x1="17.5" y1="17.5" x2="19.5" y2="19.5"/><line x1="4.5" y1="19.5" x2="6.5" y2="17.5"/><line x1="17.5" y1="6.5" x2="19.5" y2="4.5"/></svg>',
    "grid": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="{ICON_COLOR}" stroke="none"><polygon points="13,2 4,14 11,14 9,22 20,9 13,9"/></svg>',
    "discount": f'<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="2"><circle cx="12" cy="12" r="9"/><text x="12" y="16" font-size="10" text-anchor="middle" fill="{ICON_COLOR}" stroke="none" font-family="sans-serif">$</text></svg>',
}


def input_card(icon_key, label, widget_fn):
    """Renders an icon + label header inside a bordered card, then the
    input widget itself."""
    with st.container(border=True):
        st.markdown(
            f'<div class="input-card-icon">{ICONS[icon_key]}<span class="input-card-label">{label}</span></div>',
            unsafe_allow_html=True,
        )
        return widget_fn()


# ── Main-page inputs ─────────────────────────────────────────
# Uses st.number_input (not st.slider) since Streamlit sliders always
# require a fixed max_value; number_input allows genuinely uncapped
# entry except for duration, which is capped at 730 days.
st.subheader("Inputs")

row1 = st.columns(3)
row2 = st.columns(3)

with row1[0]:
    duration_days = input_card(
        "duration", "duration",
        lambda: st.number_input("", min_value=1, max_value=730, value=100, step=1, key="duration_input",
                                 help="Days simulated (max 730). Larger values mean more timesteps — solve time increases sharply.")
    )

with row1[1]:
    max_candidate_batteries = input_card(
        "battery", "max_candidate_batteries",
        lambda: st.number_input("", min_value=0, value=8, step=1, key="batteries_input")
    )

with row1[2]:
    max_candidate_chargers = input_card(
        "charger", "max_candidate_chargers",
        lambda: st.number_input("", min_value=0, value=8, step=1, key="chargers_input")
    )

with row2[0]:
    pv_max_kw = input_card(
        "pv", "PV_max_kw",
        lambda: st.number_input("", min_value=0.0, value=5000.0, step=250.0, key="pv_input")
    )

with row2[1]:
    grid_max_import = input_card(
        "grid", "GRID_max_import",
        lambda: st.number_input("", min_value=0.0, value=2000.0, step=100.0, key="grid_input")
    )

with row2[2]:
    discount_rate = input_card(
        "discount", "discount_rate",
        lambda: st.number_input("", min_value=0.0, value=0.05, step=0.01, format="%.2f", key="discount_input")
    )

# ── Run controls ──────────────────────────────────────────────
st.write("")  # small spacer
demo_mode = st.checkbox("Quick preview (faster, less accurate)", value=True)
time_limit = 30 if demo_mode else None
mip_gap = 0.05 if demo_mode else 0.001

overrides = {
    "duration": duration_days,
    "max_candidate_batteries": max_candidate_batteries,
    "max_candidate_chargers": max_candidate_chargers,
    "PV_max_kw": pv_max_kw,
    "GRID_max_import": grid_max_import,
    "discount_rate": discount_rate,
}

run_button = st.button("Run Optimisation")

# ── Run pipeline ────────────────────────────────────────────
if run_button:
    with st.spinner("Solving... this may take a moment"):
        results, df_results = run_pipeline(overrides, time_limit=time_limit, mip_gap=mip_gap)

    if results is None:
        st.error("No feasible solution found. Try widening the input ranges.")
    else:
        st.success(f"Optimisation complete — {results['status']}")
        st.session_state["results"] = results
        st.session_state["df_results"] = df_results

# ── Results display ─────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]

    st.markdown("""
    <div style="background-color:#E8EEEA; padding:25px; border-radius:8px; margin:20px 0;">
    <h2 style="text-align:center;">Optimisation Results</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total annualised cost", f"${results['total_cost']:,.0f}")
    col2.metric("Batteries selected", f"{results['number_of_batteries']:.0f}")
    col3.metric("Chargers selected", f"{results['number_of_chargers']:.0f}")
    col4.metric("Run time", f"{results['solve_time_seconds']:.1f} s")

    col5, col6, col7 = st.columns(3)
    col5.metric("PV capacity (kW)", f"{results['PV_capacity_kW']:,.1f}")
    col6.metric("BESS capacity (kWh)", f"{results['BESS_capacity_kWh']:,.1f}")
    col7.metric("Grid capacity (kW)", f"{results['grid_capacity_kW']:,.1f}")

    st.subheader("Cost breakdown")
    st.dataframe(st.session_state["df_results"])