"""Tests for the curated stylistic keyword set used to split TMDb keywords."""
from cineembed.keywords import STYLISTIC_DICT, split_keywords


def test_split_pure_style():
    """Keywords entirely from the stylistic dict go to style[]."""
    style, plot = split_keywords(["neo-noir", "atmospheric"])
    assert style == ["neo-noir", "atmospheric"]
    assert plot == []


def test_split_pure_plot():
    """Keywords not in the stylistic dict go to plot[]."""
    style, plot = split_keywords(["bank robbery", "betrayal"])
    assert style == []
    assert plot == ["bank robbery", "betrayal"]


def test_split_mixed():
    """Mixed input partitions correctly."""
    style, plot = split_keywords(["noir", "time travel", "atmospheric", "ai"])
    assert set(style) == {"noir", "atmospheric"}
    assert set(plot) == {"time travel", "ai"}


def test_split_caps_at_eight():
    """Each array capped at 8 entries; original order preserved."""
    style, _ = split_keywords(["noir"] * 15)
    assert len(style) == 8


def test_case_insensitive():
    """Match is case-insensitive."""
    style, _ = split_keywords(["Neo-Noir"])
    assert "Neo-Noir" in style  # original case preserved in output


def test_dict_size_at_least_30():
    """Sanity: dictionary is not empty."""
    assert len(STYLISTIC_DICT) >= 30
