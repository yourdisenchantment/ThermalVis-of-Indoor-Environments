# tests/test_app.py

import os

# Безголовый backend для matplotlib (до импортов pyplot)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pytest

from thermalvis_of_indoor_environments.app import run_visualization


def test_single_set_smoke():
    fig, ax = run_visualization(20.1, 20.9, 20.5, 20.7, show=False)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_two_sets_smoke():
    fig, ax = run_visualization(
        20.1,
        20.9,
        20.5,
        20.7,
        north2=20.4,
        west2=21.2,
        south2=20.8,
        east2=21.0,
        min_gap=0.05,
        show=False,
    )
    assert fig is not None and ax is not None
    plt.close(fig)


# Примеры из README - два набора (t=2)
TWO_SET_CASES = [
    (
        "Temp 1",
        dict(
            n1=20.1,
            w1=20.0,
            s1=20.4,
            e1=18.8,
            n2=20.2,
            w2=20.0,
            s2=20.5,
            e2=20.0,
            min_gap=0.05,
            alpha=0.4,
        ),
    ),
    (
        "Phi 1",
        dict(
            n1=51.0,
            w1=50.0,
            s1=51.0,
            e1=50.0,
            n2=52.0,
            w2=50.0,
            s2=52.0,
            e2=50.0,
            min_gap=0.01,
            alpha=0.4,
        ),
    ),
    (
        "Temp 2",
        dict(
            n1=25.2,
            w1=24.2,
            s1=25.4,
            e1=24.5,
            n2=24.8,
            w2=24.0,
            s2=25.0,
            e2=25.9,
            min_gap=0.05,
            alpha=0.4,
        ),
    ),
    (
        "Phi 2",
        dict(
            n1=54.0,
            w1=52.0,
            s1=53.0,
            e1=47.0,
            n2=53.0,
            w2=52.0,
            s2=53.0,
            e2=52.0,
            min_gap=0.01,
            alpha=0.4,
        ),
    ),
    (
        "Temp 3",
        dict(
            n1=23.3,
            w1=23.3,
            s1=24.0,
            e1=22.5,
            n2=24.0,
            w2=22.8,
            s2=24.3,
            e2=23.1,
            min_gap=0.05,
            alpha=0.4,
        ),
    ),
    (
        "Phi 3",
        dict(
            n1=39.0,
            w1=39.0,
            s1=35.0,
            e1=41.0,
            n2=39.0,
            w2=40.0,
            s2=35.0,
            e2=39.0,
            min_gap=0.01,
            alpha=0.4,
        ),
    ),
    (
        "Temp 4",
        dict(
            n1=22.4,
            w1=22.0,
            s1=22.4,
            e1=22.2,
            n2=22.8,
            w2=22.1,
            s2=22.8,
            e2=22.6,
            min_gap=0.05,
            alpha=0.4,
        ),
    ),
    (
        "Phi 4",
        dict(
            n1=42.0,
            w1=41.0,
            s1=41.0,
            e1=41.0,
            n2=45.0,
            w2=41.0,
            s2=43.0,
            e2=40.0,
            min_gap=0.01,
            alpha=0.4,
        ),
    ),
]


@pytest.mark.parametrize("label, case", TWO_SET_CASES, ids=[c[0] for c in TWO_SET_CASES])
def test_examples_two_sets(label, case):
    fig, ax = run_visualization(
        case["n1"],
        case["w1"],
        case["s1"],
        case["e1"],
        north2=case["n2"],
        west2=case["w2"],
        south2=case["s2"],
        east2=case["e2"],
        min_gap=case["min_gap"],
        alpha=case["alpha"],
        show=False,
    )
    assert fig is not None and ax is not None
    plt.close(fig)


# Примеры из README - один набор (берём первый набор из каждой пары)
SINGLE_SET_CASES = [
    ("Temp 1 (single)", dict(n=20.1, w=20.0, s=20.4, e=18.8)),
    ("Phi 1 (single)", dict(n=51.0, w=50.0, s=51.0, e=50.0)),
    ("Temp 2 (single)", dict(n=25.2, w=24.2, s=25.4, e=24.5)),
    ("Phi 2 (single)", dict(n=54.0, w=52.0, s=53.0, e=47.0)),
    ("Temp 3 (single)", dict(n=23.3, w=23.3, s=24.0, e=22.5)),
    ("Phi 3 (single)", dict(n=39.0, w=39.0, s=35.0, e=41.0)),
    ("Temp 4 (single)", dict(n=22.4, w=22.0, s=22.4, e=22.2)),
    ("Phi 4 (single)", dict(n=42.0, w=41.0, s=41.0, e=41.0)),
]


@pytest.mark.parametrize("label, case", SINGLE_SET_CASES, ids=[c[0] for c in SINGLE_SET_CASES])
def test_examples_single_set(label, case):
    fig, ax = run_visualization(case["n"], case["w"], case["s"], case["e"], show=False)
    assert fig is not None and ax is not None
    plt.close(fig)
