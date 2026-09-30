# Dataset catalog and preparation

The repository tracks prepared datasets in `data/base/`, row-count variants in
`data/standardized/`, and generated column summaries in
`data/standardized/summaries/`. These tracked files are sufficient to run the
benchmark from a fresh clone.

## Preparation process

Each prepared table in `data/base/` was derived from the upstream source linked
in the catalog below as follows:

1. Read source fields as text, preserving their stored representation.
2. Convert configured date/time columns to a month feature and remove the
   source date/time column.
3. Normalize Metro's `holiday="None"` values to `No Holiday` (48,143 values in
   the source snapshot).
4. Place the configured outcome in the final column.
5. Shuffle rows with `random_state=42` and reset the row index.

Generate standardized variants and summaries with:

```bash
uv run python scripts/standardize_csvs.py --sizes 100 500 1000 10000
uv run python scripts/make_summaries.py --rewrite
```

Each dataset provides 100-, 500-, 1,000-, and 10,000-row variants.
`scripts/make_summaries.py` derives each summary JSON from its standardized CSV.
The summary records the dataset filename, final-column target, inferred column
types, feature kinds, and sampled unique counts.

## Dataset catalog

| Dataset | Source | Prepared file | Source rows | Outcome | Date-to-month mapping |
|---|---|---|---:|---|---|
| AI4I 2020 Predictive Maintenance | [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | `ai4i2020.csv` | 10,000 | `Machine failure` | — |
| Bike Sharing | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | `bike_sharing.csv` | 17,379 | `cnt` | — |
| California Housing | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | `california_housing.csv` | 20,640 | `target` | — |
| KC Housing | [Kaggle: House Sales Prediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) | `kc_housing.csv` | 21,613 | `price` | — |
| Metro Interstate Traffic Volume | [UCI](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | `metro_interstate_traffic_volume.csv` | 48,204 | `traffic_volume` | `date_time` → `month` |
| Steel Industry Energy Consumption | [UCI](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) | `steel_industry_data.csv` | 35,040 | `Usage_kWh` | `date` → `month` |

## License and attribution

Prepared copies are modified from their sources as described above;
when redistributing them, preserve the applicable attribution and identify the
changes. A missing or unknown license is not permission to redistribute.

| Dataset | Upstream license or rights status | Attribution and provenance |
|---|---|---|
| AI4I 2020 Predictive Maintenance | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *AI4I 2020 Predictive Maintenance Dataset* (2020), UCI Machine Learning Repository, [DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). |
| Bike Sharing | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Hadi Fanaee-T, *Bike Sharing* (2013), UCI Machine Learning Repository, [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). |
| California Housing | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Derived from the 1990 U.S. Census and obtained through StatLib; cite R. Kelley Pace and Ronald Barry, “Sparse Spatial Autoregressions,” *Statistics & Probability Letters* 33 (1997), 291–297. |
| KC Housing | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) | *House Sales in King County, USA*, uploaded by `harlfoxem`; preserve the [Kaggle source page](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) as provenance. |
| Metro Interstate Traffic Volume | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | John Hogue, *Metro Interstate Traffic Volume* (2019), UCI Machine Learning Repository, [DOI 10.24432/C5X60B](https://doi.org/10.24432/C5X60B). |
| Steel Industry Energy Consumption | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Sathishkumar V E, Changsun Shin, and Yongyun Cho, *Steel Industry Energy Consumption* (2021), UCI Machine Learning Repository, [DOI 10.24432/C52G8C](https://doi.org/10.24432/C52G8C). |

## Prepared schemas

### AI4I 2020 Predictive Maintenance

```text
UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], TWF, HDF, PWF, OSF,
RNF, Machine failure
```

### Bike Sharing

```text
instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit,
temp, atemp, hum, windspeed, casual, registered, cnt
```

### California Housing

```text
MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
Longitude, target
```

### KC Housing

```text
Unnamed: 0, id, date, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built,
yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15, price
```

### Metro Interstate Traffic Volume

```text
holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
weather_description, month, traffic_volume
```

### Steel Industry Energy Consumption

```text
month, Lagging_Current_Reactive.Power_kVarh,
Leading_Current_Reactive_Power_kVarh, CO2(tCO2),
Lagging_Current_Power_Factor, Leading_Current_Power_Factor, NSM, WeekStatus,
Day_of_week, Load_Type, Usage_kWh
```
