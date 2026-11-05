import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from simulator import simulate

ROOT = Path(__file__).parents[1]


@pytest.fixture
def covariates(tmp_path):
    prefix = tmp_path / "cov"
    subprocess.run([sys.executable, ROOT / "make_covariates.py", prefix, "--n-facilities", "3", "--n-years", "2"], check=True)
    # chap passes the covariates with an index column
    df = pd.read_csv(prefix.with_suffix(".csv"))
    df.to_csv(prefix.with_suffix(".csv"))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"user_option_values": {"seed": 3}}))
    return prefix.with_suffix(".csv"), prefix.with_suffix(".geojson"), config


def run(tmp_path, covariates_file, polygons, config, name):
    out = tmp_path / name
    simulate(str(covariates_file), str(out), str(config), str(polygons))
    return pd.read_csv(out).sort_values(["location", "time_period"]).reset_index(drop=True)


def test_continuing_matches_one_run(tmp_path, covariates):
    covariates_file, polygons, config = covariates
    full = run(tmp_path, covariates_file, polygons, config, "full.csv")
    assert full["disease_cases"].notna().any()

    first_year = full[full["time_period"] < "2016-01"]
    rest = pd.read_csv(covariates_file, index_col=0)
    rest = rest[rest["time_period"] >= "2016-01"]
    continued_input = tmp_path / "continued_input.csv"
    pd.concat([first_year, rest]).to_csv(continued_input)
    continued = run(tmp_path, continued_input, polygons, config, "continued.csv")
    pd.testing.assert_series_equal(full["disease_cases"], continued["disease_cases"])


def test_observed_cases_are_kept(tmp_path, covariates):
    covariates_file, polygons, config = covariates
    df = pd.read_csv(covariates_file, index_col=0)
    df["disease_cases"] = float("nan")
    df.loc[df.index[0], "disease_cases"] = 12345.0
    observed = tmp_path / "observed.csv"
    df.to_csv(observed)
    out = run(tmp_path, observed, polygons, config, "out.csv")
    assert 12345.0 in out["disease_cases"].to_numpy()


def test_unknown_option_fails(tmp_path, covariates):
    covariates_file, polygons, config = covariates
    config.write_text(yaml.safe_dump({"user_option_values": {"anual_incidence": 0.2}}))
    with pytest.raises(ValueError, match="unknown simulation options"):
        run(tmp_path, covariates_file, polygons, config, "out.csv")


def test_unreported_months_are_missing(tmp_path, covariates):
    covariates_file, polygons, config = covariates
    config.write_text(yaml.safe_dump({"user_option_values": {"seed": 3, "completeness_mean": 0.5}}))
    out = run(tmp_path, covariates_file, polygons, config, "out.csv")
    assert 0 < out["disease_cases"].isna().sum() < len(out)


def test_intervention_window_can_wrap_the_new_year(tmp_path, covariates):
    covariates_file, polygons, config = covariates

    def total(options):
        config.write_text(yaml.safe_dump({"user_option_values": {"seed": 3, "completeness_mean": 0.999, **options}}))
        out = run(tmp_path, covariates_file, polygons, config, "out.csv")
        return out.groupby(out["time_period"].str[5:7])["disease_cases"].sum()

    window = {"intervention_first_month": 11, "intervention_last_month": 2}
    without = total(window)
    with_intervention = total({**window, "intervention_effect": 0.9})
    assert (with_intervention[["11", "12", "01", "02"]] < without[["11", "12", "01", "02"]]).all()
    assert (with_intervention[["05", "06", "07"]] == without[["05", "06", "07"]]).all()
