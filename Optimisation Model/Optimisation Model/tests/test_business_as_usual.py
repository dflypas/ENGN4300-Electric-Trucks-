
import pytest

from business_as_usual import business_as_usual, replacement_annualisation


def make_bau(years=None, prices=None, diesel_used=None, discount_rate=0.05, project_lifetime=10):
    """Run BAU with a small two-truck, three-route example."""
    if years is None:
        years = [0]
    if prices is None:
        prices = {0: 2.0}

    if diesel_used is None:
        diesel_used = {1: 100, 2: 150, 3: 200}

    return business_as_usual(
        routes=[1, 2, 3],
        route_diesel_used=diesel_used,
        route_year={1: 0, 2: 0, 3: 0},
        diesel_price_t=prices,
        trucks=[1, 2],
        years=years,
        trucks_in_year={y: [1, 2] for y in years},
        diesel_capex=100000,
        diesel_lifetime=10,
        diesel_om=0.05,
        discount_rate=discount_rate,
        project_lifetime=project_lifetime,
    )


def test_v44_diesel_consumption():
    result = make_bau()
    assert result.loc[0, "diesel_used_L"] == 450


def test_v45_annualised_diesel_capex():
    result = make_bau()
    crf = 0.05 * 1.05**10 / (1.05**10 - 1)
    expected = 2 * 100000 * crf
    assert result.loc[0, "annualised_capex"] == pytest.approx(expected)


def test_v46_diesel_om_cost():
    result = make_bau()
    assert result.loc[0, "OM_cost"] == 100000 * 0.05 * 2
    assert result.loc[0, "OM_cost"] == 10000


def test_v47_total_cost():
    result = make_bau()
    row = result.iloc[0]
    expected = (
        row["operation_cost"]
        + row["annualised_capex"]
        + row["replacement_cost"]
        - row["residual_value"]
        + row["OM_cost"]
    )
    assert row["total_cost"] == pytest.approx(expected)


def test_v48_diesel_prices_by_year():
    result = make_bau(years=[0, 1], prices={0: 2.0, 1: 2.5})
    assert result["diesel_price"].tolist() == [2.0, 2.5]
    assert result["operation_cost"].tolist() == pytest.approx([450 * 2.0, 450 * 2.5])


def test_v49_all_trucks_diesel():
    result = make_bau()
    assert result.loc[0, "number_of_diesel_trucks"] == 2
    assert result.loc[0, "number_of_electric_trucks"] == 0


def test_v50_no_electric_infrastructure():
    result = make_bau()
    assert (result["number_of_electric_trucks"] == 0).all()
    # BAU does not return PV, BESS, battery or charger quantities.
    # Their absence is checked here, rather than assuming they equal zero.
    for column in ["PV_capacity_kW", "BESS_capacity_kWh", "number_of_batteries", "number_of_chargers"]:
        assert column not in result.columns
