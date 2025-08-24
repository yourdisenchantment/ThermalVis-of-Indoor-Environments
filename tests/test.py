# tests/test.py

from thermalvis_of_indoor_environments.app import run_visualization


def test_single_set_smoke():
    fig, ax = run_visualization(20.1, 20.9, 20.5, 20.7, show=False)
    assert fig is not None and ax is not None


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
