"""Generate synthetic health facility covariates for ``chap simulate``.

Writes ``<out>.csv`` with ``time_period``, ``location``, ``rainfall``, ``mean_temperature`` and ``population`` for
every facility and period (month or ISO week), and ``<out>.geojson`` with a small square polygon around each facility, which the
simulator uses for the facility's coordinates.

Facilities are placed in clusters (think districts) inside a bounding box. Rainfall has one rainy season per year,
optionally wetter towards one edge of the box; temperature follows an annual cycle. All of it is set by the
arguments below; run with ``--help`` to list them.
"""

import argparse
import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

POLYGON_HALF_SIZE_DEG = 0.02


def place_facilities(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    """Draw facility coordinates around randomly placed cluster centres, with clusters of random size."""
    lat_min, lat_max, lon_min, lon_max = args.bbox
    centres = np.column_stack([rng.uniform(lat_min, lat_max, args.n_clusters), rng.uniform(lon_min, lon_max, args.n_clusters)])
    cluster = rng.choice(args.n_clusters, size=args.n_facilities, p=rng.dirichlet(np.full(args.n_clusters, 2.0)))
    spread = args.cluster_spread_deg
    lat = np.clip(centres[cluster, 0] + rng.normal(0, spread, args.n_facilities), lat_min, lat_max)
    lon = np.clip(centres[cluster, 1] + rng.normal(0, spread, args.n_facilities), lon_min, lon_max)
    ids = [f"facility_{i + 1:05d}" for i in range(args.n_facilities)]
    return pd.DataFrame({"location": ids, "lat": lat, "lon": lon})


def make_calendar(args: argparse.Namespace) -> pd.DataFrame:
    """The periods to generate: chap label, calendar year and position in the year in months (1 = mid-January).

    Weekly periods are ISO weeks, placed by their middle day; a week in the middle of a month gets that month's
    number, so the same climate arguments describe monthly and weekly data.
    """
    if args.period == "month":
        months = pd.period_range(args.start, periods=12 * args.n_years, freq="M")
        return pd.DataFrame({"label": months.strftime("%Y-%m"), "year": months.year, "month": months.month.astype(float), "per_year": 12.0})
    start = date.fromisoformat(args.start + "-01")
    end = date(start.year + args.n_years, start.month, 1)
    monday = start - timedelta(days=start.weekday())
    rows = []
    while monday < end:
        mid = monday + timedelta(days=3)
        position = mid.month + (mid.day - 0.5) / calendar.monthrange(mid.year, mid.month)[1] - 0.5
        rows.append({"label": monday.strftime("%G-W%V"), "year": mid.year, "month": position, "per_year": 365.25 / 7})
        monday += timedelta(days=7)
    return pd.DataFrame(rows)


def rainfall(lat: np.ndarray, cal: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Rainfall (mm per period) with one rainy season per year and year-to-year anomalies in total and timing."""
    lat_mid = (args.bbox[0] + args.bbox[1]) / 2
    annual_total = np.clip(args.annual_rainfall + args.rainfall_gradient * (lat - lat_mid), 0.1 * args.annual_rainfall, None)
    years = cal["year"].to_numpy()
    month_of_year = cal["month"].to_numpy()
    # Weekly rainfall is noisier than monthly.
    noise_shape = 4.0 if cal["per_year"].iloc[0] == 12 else 2.0
    out = np.empty((len(lat), len(cal)))
    for year in np.unique(years):
        cols = years == year
        total = annual_total * np.exp(rng.normal(0, args.rainfall_year_sd) + rng.normal(0, 0.1, len(lat)))
        peak = args.rain_peak_month + rng.normal(0, 0.3)
        # Distance in months to the peak, wrapping around the year so a season can span the new year.
        distance = (month_of_year[cols][None, :] - peak + 6) % 12 - 6
        shape = np.exp(-0.5 * (distance / args.rain_season_sd_months) ** 2) + np.zeros((len(lat), 1))
        shape /= shape.sum(axis=1, keepdims=True)
        out[:, cols] = total[:, None] * shape * rng.gamma(noise_shape, 1 / noise_shape, shape.shape)
    return out


def temperature(n: int, cal: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Mean temperature: an annual cycle, yearly anomalies, a warming trend and per-period noise."""
    phase = 2 * np.pi * (cal["month"].to_numpy() - args.hottest_month) / 12
    cycle = args.mean_temperature + args.temperature_amplitude * np.cos(phase)
    year_anomaly = {y: rng.normal(0, 0.3) for y in np.unique(cal["year"])}
    anomaly = np.array([year_anomaly[y] for y in cal["year"]])
    warming = args.warming_per_year * np.arange(len(cal)) / cal["per_year"].to_numpy()
    return (cycle + anomaly + warming)[None, :] + rng.normal(0, 0.7, (n, len(cal)))


def population(n: int, cal: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Catchment population: log-normal around the median, growing at a fixed yearly rate."""
    start = rng.lognormal(np.log(args.population_median), args.population_sd, n)
    years = np.arange(len(cal)) / cal["per_year"].to_numpy()
    return start[:, None] * (1 + args.population_growth) ** years[None, :]


def polygons(facilities: pd.DataFrame) -> dict:
    """A small square around each facility, so chap carries its coordinates to the simulator."""
    h = POLYGON_HALF_SIZE_DEG
    features = []
    for row in facilities.itertuples():
        ring = [[row.lon - h, row.lat - h], [row.lon + h, row.lat - h], [row.lon + h, row.lat + h], [row.lon - h, row.lat + h], [row.lon - h, row.lat - h]]
        features.append({"type": "Feature", "id": row.location, "properties": {"id": row.location}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": features}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("out_prefix", help="output path without extension; writes .csv and .geojson")
    parser.add_argument("--n-facilities", type=int, default=100)
    parser.add_argument("--period", choices=["month", "week"], default="month")
    parser.add_argument("--start", default="2015-01", help="first month, YYYY-MM; weekly data starts with the ISO week containing its first day")
    parser.add_argument("--n-years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bbox", type=float, nargs=4, default=[0.0, 4.0, 30.0, 36.0], metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--n-clusters", type=int, default=8, help="number of facility clusters, e.g. districts")
    parser.add_argument("--cluster-spread-deg", type=float, default=0.5, help="standard deviation of facility positions around a cluster centre")
    parser.add_argument("--annual-rainfall", type=float, default=1000.0, help="mean yearly rainfall (mm) at the middle latitude of the box")
    parser.add_argument("--rainfall-gradient", type=float, default=0.0, help="change in yearly rainfall (mm) per degree of latitude north")
    parser.add_argument("--rainfall-year-sd", type=float, default=0.2, help="standard deviation of the log yearly rainfall anomaly")
    parser.add_argument("--rain-peak-month", type=float, default=7.0, help="month of peak rainfall, 1-12")
    parser.add_argument("--rain-season-sd-months", type=float, default=1.5, help="width of the rainy season in months")
    parser.add_argument("--mean-temperature", type=float, default=25.0)
    parser.add_argument("--temperature-amplitude", type=float, default=3.0, help="half the difference between the hottest and coldest month")
    parser.add_argument("--hottest-month", type=float, default=3.0, help="month of the highest mean temperature, 1-12")
    parser.add_argument("--warming-per-year", type=float, default=0.03)
    parser.add_argument("--population-median", type=float, default=12000.0, help="median catchment population")
    parser.add_argument("--population-sd", type=float, default=0.6, help="standard deviation of the log catchment population")
    parser.add_argument("--population-growth", type=float, default=0.03, help="yearly population growth rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cal = make_calendar(args)
    facilities = place_facilities(args, rng)
    n = len(facilities)
    values = {
        "rainfall": rainfall(facilities["lat"].to_numpy(), cal, args, rng),
        "mean_temperature": temperature(n, cal, args, rng),
        "population": population(n, cal, args, rng),
    }
    df = pd.DataFrame(
        {
            "time_period": np.tile(cal["label"].to_numpy(), n),
            "location": np.repeat(facilities["location"].to_numpy(), len(cal)),
            **{name: v.ravel().round(2) for name, v in values.items()},
        }
    )
    out = Path(args.out_prefix)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".geojson").write_text(json.dumps(polygons(facilities)))
    print(f"{n} facilities x {len(cal)} {args.period}s -> {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
