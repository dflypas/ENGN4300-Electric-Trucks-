"""Verification tests V25-V27 and V44-V51 for DairyDrive Electric."""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / 'business_as_usual.py').exists():
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from business_as_usual import business_as_usual, replacement_annualisation


def inputs():
    return dict(routes=[0, 1, 2], route_diesel_used={0: 100, 1: 150, 2: 200},
                route_year={0: 0, 1: 0, 2: 0}, diesel_price_t={0: 2.0},
                trucks=[0, 1], years=[0], trucks_in_year={0: [0, 1]},
                diesel_capex=100000, diesel_lifetime=5, diesel_om=0.05,
                discount_rate=0.05, project_lifetime=10)


def crf(rate=0.05, years=10):
    # Independent annuity-present-value calculation.
    return 1 / sum((1 + rate) ** -year for year in range(1, years + 1))


def replacement(lifetime, years=10):
    return sum(100000 / 1.05 ** year for year in range(lifetime, years, lifetime)) * crf()


def optimiser_expression(name):
    # Evaluate the real optimiser arithmetic without solving a Gurobi model.
    tree = ast.parse((ROOT / 'optimiser.py').read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optimiser')
    matches = [n.value for n in ast.walk(func) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1
    return compile(ast.Expression(matches[0]), 'optimiser.py', 'eval')


@pytest.mark.parametrize('lifetime,years', [(15, 10), (10, 10), (5, 10), (3, 10), (4, 12)])
def test_v25_replacement_annualisation(lifetime, years):
    expected = sum(100000 / 1.05 ** t for t in range(lifetime, years, lifetime)) * crf(0.05, years)
    assert replacement_annualisation(100000, years, 0.05, lifetime) == pytest.approx(expected, abs=0.01)
    if lifetime == 5 and years == 10:
        result, _ = business_as_usual(**inputs())
        assert result['replacement_cost'] == pytest.approx(2 * expected, abs=0.01)


@pytest.mark.parametrize('rate,years', [(0.01, 1), (0.05, 10), (0.07, 20), (0.10, 5)])
def test_v26_capital_recovery_factor(rate, years):
    data = inputs()
    data.update(discount_rate=rate, project_lifetime=years)
    result, _ = business_as_usual(**data)
    expected = crf(rate, years)
    assert result['annualised_capex'] / 200000 == pytest.approx(expected)
    actual = eval(optimiser_expression('crf'), {'__builtins__': {}},
                  dict(discount_rate=rate, project_lifetime=years))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('litres,price', [(50, 2), (0, 2), (12.5, 2.35)])
def test_v27_diesel_fuel_cost(litres, price):
    data = inputs()
    data.update(routes=[0], route_diesel_used={0: litres}, route_year={0: 0}, diesel_price_t={0: price})
    result, _ = business_as_usual(**data)
    assert result['diesel_fuel_cost'] == pytest.approx(litres * price)
    assert result['operation_cost'] == pytest.approx(litres * price)
    # Check the optimiser's actual expression, including diesel/electric selection.
    class FakeGP:
        quicksum = staticmethod(sum)
    expr = optimiser_expression('an_op_diesel')
    for electric, expected in [(0, litres * price), (1, 0)]:
        actual = eval(expr, dict(
            __builtins__={}, gp=FakeGP, routes=[0], route_diesel_used={0: litres},
            diesel_price=price, route_truck={0: 7}, z={7: electric}))
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('amounts', [(100, 150, 200), (0, 0, 0), (1.25, 2.5, 3.75)])
def test_v44_diesel_consumption(amounts):
    data = inputs()
    data['route_diesel_used'] = dict(enumerate(amounts))
    result, _ = business_as_usual(**data)
    assert result['diesel_used_L'] == pytest.approx(sum(amounts))


def test_v45_annualised_diesel_capex():
    result, _ = business_as_usual(**inputs())
    assert result['annualised_capex'] == pytest.approx(200000 * crf())


def test_v46_diesel_om_cost():
    result, _ = business_as_usual(**inputs())
    assert result['OM_cost'] == pytest.approx(10000)


def test_v47_total_cost():
    result, _ = business_as_usual(**inputs())
    expected = dict(operation_cost=900, annualised_capex=200000 * crf(),
                    replacement_cost=2 * replacement(5), OM_cost=10000)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=0.01)
    assert result['total_cost'] == pytest.approx(sum(expected.values()), abs=0.01)


def test_v48_diesel_price_changes_between_years():
    data = inputs()
    data['route_year'] = {0: 0, 1: 0, 2: 1}
    data['diesel_price_t'] = {0: 2.0, 1: 2.5}
    result, _ = business_as_usual(**data)
    assert result['diesel_fuel_cost'] == pytest.approx(100 * 2 + 150 * 2 + 200 * 2.5)


def test_v49_all_trucks_remain_diesel():
    result, _ = business_as_usual(**inputs())
    assert result['number_of_diesel_trucks'] == 2
    assert result['number_of_electric_trucks'] == 0


def test_v50_no_electric_infrastructure():
    result, _ = business_as_usual(**inputs())
    for name in ['number_of_batteries', 'number_of_chargers', 'BESS_capacity_kWh', 'PV_capacity_kW']:
        assert result[name] == 0

