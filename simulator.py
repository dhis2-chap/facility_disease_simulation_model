"""Simulate monthly disease cases at health facilities from rainfall, temperature and population.

Expected cases for facility i in month t:

    mu = population / 12 * annual_incidence / season_norm_i
         * facility_effect_i * season_it * heat_it * year_effect_t
         * global_trend_t * facility_trend_it * latent_it * intervention_t

Reported cases are negative binomial around mu, reduced in stockout months, occasionally multiplied by a
data-entry outlier, and missing when the facility did not report. See README.md for each term.

Every random draw depends only on the seed, the facility and the period (and, for the spatial facility effect,
on the facility's coordinates), so rows that already have ``disease_cases`` can be kept and the remaining rows
simulated with the same result as one run over all periods.
"""

import json
import sys
import zlib

import numpy as np
import pandas as pd
import yaml

DEFAULTS = {
    "seed": 0,
    "annual_incidence": 0.15,
    "facility_sd": 0.8,
    "spatial_fraction": 0.5,
    "spatial_range_km": 75.0,
    "rainfall_lag_min": 1,
    "rainfall_lag_max": 2,
    "facility_lag_shift_max": 1,
    "rainfall_half_saturation": 80.0,
    "dry_season_level": 0.15,
    "dry_season_sd": 0.3,
    "season_amplitude_sd": 0.25,
    "temperature_threshold": 32.0,
    "temperature_effect": 0.1,
    "year_sd": 0.15,
    "annual_trend": 0.03,
    "global_trend_sd": 0.15,
    "facility_trend_sd": 0.08,
    "latent_ar": 0.7,
    "latent_sd": 0.3,
    "intervention_effect": 0.0,
    "intervention_start_year": 0,
    "intervention_first_month": 7,
    "intervention_last_month": 10,
    "dispersion": 5.0,
    "stockout_rate": 0.01,
    "stockout_factor": 0.3,
    "outlier_rate": 0.002,
    "outlier_factor": 10.0,
    "completeness_mean": 0.9,
    "completeness_concentration": 10.0,
    "outage_rate": 0.004,
    "outage_max_months": 18,
    "late_start_fraction": 0.05,
    "late_start_max_months": 96,
}
N_SPATIAL_FEATURES = 500
KM_PER_DEGREE = 111.0


def rng_for(seed: int, *keys: str) -> np.random.Generator:
    """A generator that depends only on the seed and the keys, not on what else is in the dataset."""
    return np.random.default_rng([seed, *(zlib.crc32(key.encode()) for key in keys)])


def facility_coordinates(polygons_file: str) -> dict[str, tuple[float, float]]:
    """Centroid (lat, lon) of each facility's polygon, keyed by location id."""
    with open(polygons_file) as f:
        features = json.load(f)["features"]
    coords = {}
    for feature in features:
        ring = np.array(feature["geometry"]["coordinates"][0])
        coords[str(feature["id"])] = (float(ring[:, 1].mean()), float(ring[:, 0].mean()))
    return coords


def spatial_field(seed: int, lat: float, lon: float, range_km: float) -> float:
    """A smooth unit-variance random field over space, as a fixed function of the coordinates.

    Random Fourier features of a squared-exponential kernel, so a facility's value does not depend on which other
    facilities are in the dataset.
    """
    rng = rng_for(seed, "spatial_field")
    frequencies = rng.normal(0, 1 / range_km, (N_SPATIAL_FEATURES, 2))
    phases = rng.uniform(0, 2 * np.pi, N_SPATIAL_FEATURES)
    xy = np.array([lat * KM_PER_DEGREE, lon * KM_PER_DEGREE * np.cos(np.radians(lat))])
    return float(np.sqrt(2 / N_SPATIAL_FEATURES) * np.cos(frequencies @ xy + phases).sum())


def yearly_random_walk(seed: int, sd: float, months_since_start: np.ndarray, *keys: str) -> np.ndarray:
    """A random walk with one step per year, interpolated linearly between yearly knots; 0 at the first month."""
    n_knots = int(months_since_start.max()) // 12 + 2
    steps = [rng_for(seed, *keys, str(k)).normal(0, sd) for k in range(1, n_knots)]
    knots = np.concatenate([[0.0], np.cumsum(steps)])
    return np.interp(months_since_start / 12, np.arange(n_knots), knots)


def lagged_mean(values: np.ndarray, lag_min: int, lag_max: int) -> np.ndarray:
    """Mean over lags lag_min..lag_max; before the first lagged month exists, use the earliest value."""
    lags = range(lag_min, lag_max + 1)
    return np.mean([np.concatenate([np.full(lag, values[0]), values[: len(values) - lag]]) for lag in lags], axis=0)


def reported(location: str, periods: list[str], months_since_start: np.ndarray, params: dict) -> np.ndarray:
    """Which months the facility reported: late opening, multi-month outages and random missed months."""
    seed = int(params["seed"])
    n = len(periods)
    facility = rng_for(seed, "reporting", location)
    mean, concentration = float(params["completeness_mean"]), float(params["completeness_concentration"])
    completeness = facility.beta(mean * concentration, (1 - mean) * concentration)
    opens_at = 0
    if facility.uniform() < float(params["late_start_fraction"]):
        opens_at = int(facility.integers(1, int(params["late_start_max_months"]) + 1))
    out = months_since_start >= opens_at
    for t, period in enumerate(periods):
        draw = rng_for(seed, "missing", location, period)
        if draw.uniform() > completeness:
            out[t] = False
        if draw.uniform() < float(params["outage_rate"]):
            out[t : t + int(draw.integers(1, int(params["outage_max_months"]) + 1))] = False
    return out[:n]


def simulate_facility(df: pd.DataFrame, location: str, coords: tuple[float, float], params: dict) -> pd.Series:
    """Simulate the missing ``disease_cases`` for one facility, with rows sorted by time period."""
    seed = int(params["seed"])
    periods = [str(p) for p in df["time_period"]]
    n = len(df)
    if n < 12:
        raise ValueError(f"facility {location} has {n} periods; at least 12 are needed to normalize the season")
    years = np.array([int(p[:4]) for p in periods])
    months = np.array([int(p[5:7]) for p in periods])
    months_since_start = (years - years[0]) * 12 + months - months[0]
    facility = rng_for(seed, "facility", location)

    total_sd = float(params["facility_sd"])
    spatial_sd = total_sd * np.sqrt(float(params["spatial_fraction"]))
    own_sd = total_sd * np.sqrt(1 - float(params["spatial_fraction"]))
    log_facility = spatial_sd * spatial_field(seed, *coords, float(params["spatial_range_km"])) + facility.normal(0, own_sd)

    # Season: saturating in lagged rainfall, with a facility-specific lag, dry-season level and yearly amplitude.
    shift = int(facility.integers(0, int(params["facility_lag_shift_max"]) + 1))
    rain = lagged_mean(
        df["rainfall"].to_numpy(dtype=float), int(params["rainfall_lag_min"]) + shift, int(params["rainfall_lag_max"]) + shift
    )
    dry = float(params["dry_season_level"]) * np.exp(facility.normal(0, float(params["dry_season_sd"])))
    amplitude = np.exp([rng_for(seed, "amplitude", location, str(y)).normal(0, float(params["season_amplitude_sd"])) for y in years])
    seasonal = dry + amplitude * rain / (rain + float(params["rainfall_half_saturation"]))
    # annual_incidence is the incidence in the facility's first year, before the facility effect.
    seasonal = seasonal / seasonal[:12].mean()

    excess_heat = np.clip(df["mean_temperature"].to_numpy(dtype=float) - float(params["temperature_threshold"]), 0, None)
    heat = np.exp(-float(params["temperature_effect"]) * excess_heat)
    year_effect = np.exp([rng_for(seed, "year", str(y)).normal(0, float(params["year_sd"])) for y in years])
    global_trend = np.exp(
        float(params["annual_trend"]) * months_since_start / 12
        + yearly_random_walk(seed, float(params["global_trend_sd"]), months_since_start, "global_trend")
    )
    facility_trend = np.exp(yearly_random_walk(seed, float(params["facility_trend_sd"]), months_since_start, "facility_trend", location))
    first, last = int(params["intervention_first_month"]), int(params["intervention_last_month"])
    # The intervention window may wrap around the new year, e.g. November to February.
    in_window = (months >= first) & (months <= last) if first <= last else (months >= first) | (months <= last)
    intervention = np.where(
        in_window & (years >= int(params["intervention_start_year"])), 1 - float(params["intervention_effect"]), 1.0
    )

    ar = float(params["latent_ar"])
    innovation_sd = float(params["latent_sd"]) * np.sqrt(1 - ar**2)
    latent = np.empty(n)
    for t, period in enumerate(periods):
        draw = rng_for(seed, "latent", location, period)
        latent[t] = draw.normal(0, float(params["latent_sd"])) if t == 0 else ar * latent[t - 1] + draw.normal(0, innovation_sd)

    mu = (
        df["population"].to_numpy(dtype=float) / 12 * float(params["annual_incidence"])
        * np.exp(log_facility) * seasonal * heat * year_effect * global_trend * facility_trend * np.exp(latent) * intervention
    )

    # Stockouts: a stockout starts in a month with probability stockout_rate and lasts 1-3 months.
    stocked = np.ones(n)
    for t, period in enumerate(periods):
        draw = rng_for(seed, "stockout", location, period)
        if draw.uniform() < float(params["stockout_rate"]):
            stocked[t : t + draw.integers(1, 4)] = float(params["stockout_factor"])

    is_reported = reported(location, periods, months_since_start, params)
    k = float(params["dispersion"])
    cases = df["disease_cases"].to_numpy(dtype=float).copy()
    for t, period in enumerate(periods):
        if not np.isnan(cases[t]):
            continue
        if not is_reported[t]:
            continue
        draw = rng_for(seed, "cases", location, period)
        mean = mu[t] * stocked[t]
        value = float(draw.negative_binomial(k, k / (k + mean)))
        if draw.uniform() < float(params["outlier_rate"]):
            value = np.round(value * float(params["outlier_factor"]))
        cases[t] = value
    return pd.Series(cases, index=df.index)


def simulate(covariates_file: str, out_file: str, config_file: str, polygons_file: str) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
    options = config.get("user_option_values") or {}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown simulation options: {sorted(unknown)}")
    params = DEFAULTS | options

    df = pd.read_csv(covariates_file, index_col=0).sort_values(["location", "time_period"])
    if "disease_cases" not in df.columns:
        df["disease_cases"] = np.nan
    coords = facility_coordinates(polygons_file)
    missing = sorted(set(df["location"].astype(str)) - set(coords))
    if missing:
        raise ValueError(f"no polygon for locations {missing[:5]}{'...' if len(missing) > 5 else ''}")
    simulated = [
        simulate_facility(group, str(location), coords[str(location)], params)
        for location, group in df.groupby("location")
    ]
    df["disease_cases"] = pd.concat(simulated)
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "simulate":
        simulate(*args)
    else:
        raise NotImplementedError(f"This is a simulation-only model; '{command}' is not supported")
