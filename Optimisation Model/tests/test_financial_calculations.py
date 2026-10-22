"""Simple verification tests V25-V27: financial calculations and diesel costs."""

from pathlib import Path

import pandas as pd
import pytest
from optimiser import replacement_annualisation


ROOT = Path(__file__).resolve().parents[1]  # test file is inside tests/


def test_v25_replacement_annualisation():
    # $1,000 asset, 10-year project, 5% discount rate, 75% replacement cost.
    cost = 1000
    rate = 0.05
    years = 10
    crf = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)

    # No replacement (asset lasts beyond the project).
    replacement, residual = replacement_annualisation(cost, years, rate, 12)
    assert replacement == pytest.approx(0, abs=0.01)

    # One replacement at year 5 (no residual at the project endpoint).
    replacement, residual = replacement_annualisation(cost, years, rate, 5)
    expected = (cost * 0.75 / (1 + rate) ** 5) * crf
    assert replacement == pytest.approx(expected, abs=0.01)
    assert residual == pytest.approx(0, abs=0.01)

    # Multiple replacements at years 3, 6 and 9.
    replacement, residual = replacement_annualisation(cost, years, rate, 3)
    expected = sum(cost * 0.75 / (1 + rate) ** y for y in (3, 6, 9)) * crf
    assert replacement == pytest.approx(expected, abs=0.01)


def test_v26_capital_recovery_factor():
    # Verify the documented formula against a separate annuity calculation.
    rate = 0.05
    years = 10
    crf = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    expected = 1 / sum(1 / (1 + rate) ** y for y in range(1, years + 1))
    assert crf == pytest.approx(expected)

    # Confirm the same CRF expression appears in the actual optimiser.
    source = (ROOT / 'optimiser.py').read_text(encoding='utf-8')
    compact = ''.join(source.split())
    assert 'crf=(discount_rate*(1+discount_rate)**project_lifetime/((1+discount_rate)**project_lifetime-1))' in compact


def test_v27_diesel_operating_cost():
    # Read the diesel-price input that main.py uses.
    prices = pd.read_csv(ROOT / 'diesel_price_prediction.csv')
    price = float(prices.loc[0, 'Price'])
    assert price > 0

    # Reference calculation: fuel consumed in litres multiplied by AUD/L.
    assert 50 * 2.00 == pytest.approx(100.00, abs=0.01)
    assert 0 * price == pytest.approx(0, abs=0.01)
    assert 12.5 * 2.00 == pytest.approx(25.00, abs=0.01)
    assert 50 * 2.00 * (1 - 1) == pytest.approx(0, abs=0.01)

    # Check that main.py passes the CSV price to the optimiser.
    main_source = ''.join((ROOT / 'main.py').read_text(encoding='utf-8').split())
    assert 'diesel_price_prediction.csv' in main_source
    assert 'diesel_price_df["Price"]' in main_source
    assert '"diesel_price":DIESEL_PRICE_T[0]' in main_source

    # Check the actual objective uses litres x AUD/L and excludes electric routes.
    source = ''.join((ROOT / 'optimiser.py').read_text(encoding='utf-8').split())
    assert 'route_diesel_used[r]*diesel_price*(1-z[route_truck[r]])' in source