import requests
import itertools
import json
import os
import time
import xarray as xr
import numpy as np
import sys

if len(sys.argv) <= 1:
    print("No mode specified, defaulting to 'deterministic'. To specify mode, run: python deterministic.py [deterministic|probabilistic]")
    sys.exit(1)
    
mode = sys.argv[1]
assert mode in ["deterministic", "probabilistic"], "Mode must be 'deterministic' or 'probabilistic'"
print(f"Running in {mode} mode...")

BASE = f"https://researchsites.withgoogle.com/weatherbench/{mode}"
UPDATE_URL = BASE + "/_dash-update-component"
LAYOUT_URL = BASE + "/_dash-layout"

OUTPUT_DIR = "scraping/scores"
OUTPUT_DIR_RAW = "scraping/raw"
VARIABLES = ['10m U Component of Wind', '10m V Component of Wind', '10m Wind Speed', '24h Precipitation', '2m Temperature', '6h Precipitation', 'Geopotential', 'Sea Level Pressure', 'Specific Humidity', 'Temperature', 'U Component of Wind', 'V Component of Wind', 'Wind Speed']
METRICS = {
    "probabilistic": ['CRPS', 'Spread/Skill', 'Unbiased Spread/Skill', 'Unbiased Mean RMSE', 'Mean RMSE'],
    "deterministic": ['ACC', 'Bias', 'Forecast Activity', 'MSE', 'RMSE', 'SEEPS']
}[mode]
LEVELS = [500, 700, 850]

SURFACE_VARIABLES = [
    "10m U Component of Wind",
    "10m V Component of Wind",
    "10m Wind Speed",
    "24h Precipitation",
    "2m Temperature",
    "6h Precipitation",
    "Sea Level Pressure",
]
FILENAMES = {
    "probabilistic": {
        "FGN (oper.) vs Analysis": "fgn_oper_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) vs ERA5": "fgn_oper_vs_era5_1440x721_2022.nc",
        "GenCast (oper.) vs Analysis": "gencast_oper_vs_hres_t0_1440x721_2022.nc",
        "GenCast (oper.) vs ERA5": "gencast_oper_vs_era5_1440x721_2022.nc",
        "IFS ENS vs Analysis": "ifs_ens_vs_hres_t0_1440x721_2022.nc",
        "IFS ENS vs ERA5": "ifs_ens_vs_era5_1440x721_2022.nc",
        "Aurora (oper.) vs Analysis": "aurora_oper_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) (1st member) vs Analysis": "fgn_oper_first_member_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) (1st member) vs ERA5": "fgn_oper_first_member_vs_era5_1440x721_2022.nc",
        "FGN (oper.) (mean) vs Analysis": "fgn_oper_mean_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) (mean) vs ERA5": "fgn_oper_mean_vs_era5_1440x721_2022.nc",
    },
    "deterministic": {
        "GenCast (oper.) (1st member) vs Analysis": "gencast_oper_first_member_vs_hres_t0_1440x721_2022.nc",
        "GenCast (oper.) (1st member) vs ERA5": "gencast_oper_first_member_vs_era5_1440x721_2022.nc",
        "GenCast (oper.) (mean) vs Analysis": "gencast_oper_mean_vs_hres_t0_1440x721_2022.nc",
        "GenCast (oper.) (mean) vs ERA5": "gencast_oper_mean_vs_era5_1440x721_2022.nc",
        "GraphCast (oper.) vs Analysis": "graphcast_oper_vs_hres_t0_1440x721_2022.nc",
        "GraphCast (oper.) vs ERA5": "graphcast_oper_vs_era5_1440x721_2022.nc",
        "GraphCast vs ERA5": "graphcast_vs_era5_1440x721_2022.nc",
        "IFS ENS (1st member) vs Analysis": "ifs_ens_first_member_vs_hres_t0_1440x721_2022.nc",
        "IFS ENS (1st member) vs ERA5": "ifs_ens_first_member_vs_era5_1440x721_2022.nc",
        "IFS ENS (mean) vs Analysis": "ifs_ens_mean_vs_hres_t0_1440x721_2022.nc",
        "IFS ENS (mean) vs ERA5": "ifs_ens_mean_vs_era5_1440x721_2022.nc",
        "IFS HRES vs Analysis": "ifs_hres_vs_hres_t0_1440x721_2022.nc",
        "IFS HRES vs ERA5": "ifs_hres_vs_era5_1440x721_2022.nc",
        "Pangu-Weather (oper.) vs Analysis": "pangu_weather_oper_vs_hres_t0_1440x721_2022.nc",
        "Pangu-Weather (oper.) vs ERA5": "pangu_weather_oper_vs_era5_1440x721_2022.nc",
        "Pangu-Weather vs ERA5": "pangu_weather_vs_era5_1440x721_2022.nc",
    }
}[mode]
assert len(set(FILENAMES.keys()))==len(FILENAMES), "Dataset keys in FILENAMES must be unique"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)

session = requests.Session()
print("Initializing session...")
session.get(BASE)

print("Fetching Dash layout...")
layout = session.get(LAYOUT_URL).json()

# ------------------------------------------------------------
# Template
# ------------------------------------------------------------
PAYLOAD = {
    "deterministic": {
        "output":"..graph.figure...alert-relative.is_open...alert-seeps.is_open...alert-tbd.is_open..",
        "outputs":[
            {"id":"graph","property":"figure"},
            {"id":"alert-relative","property":"is_open"},
            {"id":"alert-seeps","property":"is_open"},
            {"id":"alert-tbd","property":"is_open"}
        ],
        "inputs":[
            {"id":"variable","property":"value","value":"Temperature"},
            {"id":"metric","property":"value","value":"RMSE"},
            {"id":"level","property":"value","value":850},
            {"id":"region","property":"value","value":"Global"},
            {"id":"year","property":"value","value":"2022"},
            {"id":"resolution","property":"value","value":"240x121"},
            {"id":"relative","property":"value","value":"Absolute scores"},
            {"id":"relative_to","property":"value","value":"IFS HRES vs Analysis"},
            {"id":"markers","property":"value","value":[1]}
        ],
        "changedPropIds":[],
        "state":[
            {"id":"graph","property":"figure"}
        ]
    },
    "probabilistic": {
        "output":"..graph.figure...alert-relative.is_open...alert-seeps.is_open...alert-tbd.is_open..",
        "outputs":[
            {"id":"graph","property":"figure"},
            {"id":"alert-relative","property":"is_open"},
            {"id":"alert-seeps","property":"is_open"},
            {"id":"alert-tbd","property":"is_open"}
        ],
        "inputs":[
            {"id":"variable","property":"value","value":"Temperature"},
            {"id":"metric","property":"value","value":"CRPS"},
            {"id":"level","property":"value","value":850},
            {"id":"region","property":"value","value":"Global"},
            {"id":"year","property":"value","value":"2022"},
            {"id":"resolution","property":"value","value":"1440x721"},
            {"id":"relative","property":"value","value":"Absolute scores"},
            {"id":"relative_to","property":"value","value":"IFS ENS vs Analysis"},
            {"id":"markers","property":"value","value":[1]}
        ],
        "changedPropIds":[],
        "state":[
            {"id":"graph","property":"figure"}
        ]
    }
}[mode]

# validate PAYLOAD structure
r = session.post(UPDATE_URL, json=PAYLOAD)
r.raise_for_status()
data = r.json()
print("Sample response keys:", data.keys())

# ------------------------------------------------------------
# Looping
# ------------------------------------------------------------
datasets = {
    v: xr.Dataset() for v in FILENAMES.values()
}

def clean_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "")
        .replace("-", "_")
    )

for variable, metric, level in itertools.product(VARIABLES, METRICS, LEVELS):
    print(f"Processing: {variable}, {metric}, {level}hPa")
    payload = PAYLOAD.copy()
    payload["inputs"][0]["value"] = variable
    payload["inputs"][1]["value"] = metric
    payload["inputs"][2]["value"] = level

    try:
        # cache!
        raw_out_path = os.path.join(OUTPUT_DIR_RAW, f"{clean_name(variable)}_{clean_name(metric)}_{level}hPa.json")
        if os.path.exists(raw_out_path):
            print(f"Raw response already exists at {raw_out_path}, skipping request.")
            with open(raw_out_path, "r") as f:
                data = json.load(f)
        else:
            r = session.post(UPDATE_URL, json=payload)
            r.raise_for_status()
            data = r.json()
        
        # save raw response
        with open(raw_out_path, "w") as f:
            json.dump(data, f)

        # Extract figure traces
        fig_data = data["response"]["graph"]["figure"]["data"]

        region_value = payload["inputs"][3]["value"]
        level_value = level

        for trace in fig_data:
            dataset_key = trace["name"]
            if dataset_key not in FILENAMES:
                continue

            ds = datasets[FILENAMES[dataset_key]]

            # Lead time
            x = np.array(trace["x"], dtype=float)
            lead_time = (x * 3600).astype("timedelta64[s]")

            # Values
            y = np.array(
                [v if v is not None else np.nan for v in trace["y"]],
                dtype=np.float32,
            )

            metric_name = clean_name(metric)
            variable_name = clean_name(variable)

            data_var_name = f"{metric_name}.{variable_name}"

            # Ensure coordinates exist
            if "lead_time" not in ds.coords:
                ds.coords["lead_time"] = lead_time

            if "region" not in ds.coords:
                ds.coords["region"] = np.array([region_value])

            # SURFACE VARIABLES
            if variable in SURFACE_VARIABLES:

                da = xr.DataArray(
                    y[:, None],  # add region dim
                    dims=("lead_time", "region"),
                    coords={
                        "lead_time": lead_time,
                        "region": [region_value],
                    },
                )

                ds[data_var_name] = da

            # ATMOSPHERIC VARIABLES
            else:

                # Ensure level coordinate exists
                if "level" not in ds.coords:
                    ds.coords["level"] = np.array(LEVELS)

                # Initialize variable if not existing
                if data_var_name not in ds:

                    ds[data_var_name] = xr.DataArray(
                        np.full(
                            (len(lead_time), len(ds.level), 1),
                            np.nan,
                            dtype=np.float32,
                        ),
                        dims=("lead_time", "level", "region"),
                        coords={
                            "lead_time": lead_time,
                            "level": ds.level.values,
                            "region": [region_value],
                        },
                    )

                # Insert data into correct level slice
                ds[data_var_name].loc[
                    dict(level=level_value, region=region_value)
                ] = y

        time.sleep(0.5)  # polite delay

    except Exception as e:
        print("Error:", e)
        continue

# ------------------------------------------------------------
# Save all datasets to NetCDF
# ------------------------------------------------------------
for ds_name, ds in datasets.items():
    if ds:
        out_path = os.path.join(OUTPUT_DIR, ds_name)
        print(f"Saving {out_path}")
        ds.to_netcdf(out_path)

print("All datasets saved successfully.")