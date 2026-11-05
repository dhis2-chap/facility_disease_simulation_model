# facility_disease_simulation_model

A [chap](https://github.com/dhis2-chap/chap-core) simulation model for monthly disease cases at health facilities.
It simulates `disease_cases` from `rainfall`, `mean_temperature` and `population`, producing datasets with a known
data-generating process: facility-level heterogeneity, climate-driven seasonality, trends, noisy counts and
incomplete reporting. Use it, for example, to test how a forecasting model behaves on data of a given size and
shape. It is simulation-only: `train` and `predict` raise `NotImplementedError`.

## Usage

Generate covariates (or bring your own CSV with the three covariates and a GeoJSON with one polygon per location),
then simulate cases with `chap simulate`:

```console
uv run python make_covariates.py output/facilities --n-facilities 500 --n-years 10
chap simulate . output/facilities.csv output/simulated.csv --model-configuration-yaml config.yaml
uv run python plot_simulation.py output/simulated.csv output/simulated.png
```

`config.yaml` sets any of the options below:

```yaml
user_option_values:
  seed: 1
  facility_sd: 0.6
  intervention_effect: 0.25
  intervention_start_year: 2019
```

The output is a regular chap dataset (with a `.geojson` next to it) and can be passed to `chap eval`. Missing
values, for months a facility did not report, need chap-core with
[#848](https://github.com/dhis2-chap/chap-core/pull/848); earlier versions reject them.

## Covariates (`make_covariates.py`)

`uv run python make_covariates.py --help` lists every argument with its default.

- **Facilities** are placed in clusters (think districts) inside a bounding box (`--bbox`, `--n-clusters`,
  `--cluster-spread-deg`) and named `facility_<n>`. Each gets a small square polygon so chap carries its coordinates
  to the simulator.
- **Rainfall** has one rainy season a year (`--rain-peak-month`, `--rain-season-sd-months`) with a yearly total
  (`--annual-rainfall`) that can change with latitude (`--rainfall-gradient`), plus year-to-year anomalies in total
  and timing and monthly noise.
- **Temperature** follows an annual cycle (`--mean-temperature`, `--temperature-amplitude`, `--hottest-month`) with
  yearly anomalies, warming and monthly noise.
- **Population** is log-normal around `--population-median`, growing at `--population-growth` a year.

## Cases (`simulator.py`)

Expected cases for facility *i* in month *t*:

```
mu = population / 12 * annual_incidence / season_norm_i
     * facility_effect_i * season_it * heat_it * year_effect_t
     * global_trend_t * facility_trend_it * latent_it * intervention_t
```

| Term | Option(s) | Meaning |
|---|---|---|
| `annual_incidence`, `season_norm_i` | `annual_incidence` | Scale: cases per person per year in the facility's first year, before the facility effect |
| `facility_effect_i` | `facility_sd`, `spatial_fraction`, `spatial_range_km` | Log-normal; part of it is a smooth spatial field shared with nearby facilities |
| `season_it` | `rainfall_lag_min`, `rainfall_lag_max`, `facility_lag_shift_max`, `rainfall_half_saturation`, `dry_season_level`, `dry_season_sd`, `season_amplitude_sd` | Saturating function of mean rainfall over the lag window; the lag and dry-season level vary by facility, the rainy-season amplitude by facility and year |
| `heat_it` | `temperature_threshold`, `temperature_effect` | Transmission suppressed above the threshold |
| `year_effect_t` | `year_sd` | Shared by all facilities |
| `global_trend_t` | `annual_trend`, `global_trend_sd` | Shared by all facilities: a linear drift (e.g. wider testing) plus a smooth random walk with one step per year |
| `facility_trend_it` | `facility_trend_sd` | Each facility's own smooth yearly random walk around the global trend, with smaller steps |
| `latent_it` | `latent_ar`, `latent_sd` | AR(1) on the log scale, per facility |
| `intervention_t` | `intervention_effect`, `intervention_start_year`, `intervention_first_month`, `intervention_last_month` | A seasonal intervention such as chemoprevention; the window may wrap the new year |

Reported cases are negative binomial around `mu` (`dispersion`), multiplied by `stockout_factor` during test-kit
stockouts (`stockout_rate`, lasting 1-3 months) and occasionally by `outlier_factor` as a data-entry error
(`outlier_rate`).

Months a facility did not report are left missing:

- each facility has its own completeness, drawn from a beta distribution (`completeness_mean`,
  `completeness_concentration`), and misses months at random;
- reporting outages of 1 to `outage_max_months` months start with probability `outage_rate` per month, e.g.
  insecurity or staff loss;
- a share `late_start_fraction` of facilities open up to `late_start_max_months` after the first month.

## Continuing a simulation

Every random draw depends only on the seed, the facility, the period and the facility's coordinates. Rows that
already have `disease_cases` are kept, and simulating the rest gives the same result as one run over all periods,
as long as the file still starts at the same month (the season is normalized over each facility's first 12 months).

## Development

```console
uv run pytest
```
