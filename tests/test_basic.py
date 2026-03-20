"""Basic tests to ensure the package is properly installed."""

import importlib

import polars as pl

from eurovision_voting_bloc_party.utils import prepare_eurovision_tabular_data


def test_package_import() -> None:
    """Test that the package can be imported."""
    package_name = "eurovision_voting_bloc_party"
    module = importlib.import_module(package_name)
    assert module is not None


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    package_name = "eurovision_voting_bloc_party"
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)


def test_prepare_eurovision_tabular_data() -> None:
    """Test that contest, country, and song data are joined correctly."""
    contest_data = pl.DataFrame({"year": [2023], "host": ["United Kingdom"]})
    country_data = pl.DataFrame(
        {
            "country": ["United Kingdom", "Portugal"],
            "region": ["Western Europe", "Western Europe"],
        }
    )
    song_data = pl.DataFrame({"year": [2023], "country": ["Portugal"]})

    result = prepare_eurovision_tabular_data(
        {
            "contest": contest_data,
            "country": country_data,
            "song": song_data,
        }
    )

    assert "host_region" in result.columns
    assert "participant_region" in result.columns
    assert result["host_region"][0] == "Western Europe"
    assert result["participant_region"][0] == "Western Europe"
    assert len(result) == 1
