"""Unit tests for cmi.py."""

from hinglishlens.cmi import (
    ALL_LANG_TAGS,
    compute_cmi,
    compute_lang_distribution,
    compute_switch_points,
)


def test_compute_cmi_all_hindi():
    tags = ["HI", "HI", "HI", "HI"]
    assert compute_cmi(tags) == 0.0


def test_compute_cmi_balanced_hi_en():
    tags = ["HI", "HI", "EN", "EN"]
    # n=4, u=0, max(2,2)=2. CMI = 100 * (1 - 2/4) = 50.0
    assert compute_cmi(tags) == 50.0


def test_compute_cmi_only_independent():
    tags = ["UNIV", "NE", "OTHER"]
    assert compute_cmi(tags) == 0.0


def test_compute_cmi_empty():
    assert compute_cmi([]) == 0.0


def test_compute_switch_points():
    # HI -> EN -> HI gives 2 switches
    tags = ["HI", "UNIV", "EN", "NE", "HI"]
    assert compute_switch_points(tags) == 2


def test_compute_switch_points_no_switches():
    tags = ["HI", "HI", "UNIV", "HI"]
    assert compute_switch_points(tags) == 0


def test_compute_lang_distribution():
    tags = ["HI", "EN", "EN", "UNIV", "NE"]
    dist = compute_lang_distribution(tags)
    for tag in ALL_LANG_TAGS:
        assert tag in dist
    assert dist["HI"] == 1
    assert dist["EN"] == 2
    assert dist["NE"] == 1
    assert dist["UNIV"] == 1
    assert dist["OTHER"] == 0
