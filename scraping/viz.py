import os
import xarray as xr
import matplotlib.pyplot as plt
import sys
import numpy as np

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

FILENAMES = {
    "probabilistic": {
        "FGN (oper.) vs Analysis": "fgn_oper_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) vs ERA5": "fgn_oper_vs_era5_1440x721_2022.nc",
        "GenCast (oper.) vs Analysis": "gencast_oper_vs_hres_t0_1440x721_2022.nc",
        "GenCast (oper.) vs ERA5": "gencast_oper_vs_era5_1440x721_2022.nc",
        "IFS ENS vs Analysis": "ifs_ens_vs_hres_t0_1440x721_2022.nc",
        "IFS ENS vs ERA5": "ifs_ens_vs_era5_1440x721_2022.nc",
    },
    "deterministic": {
        "Aurora (oper.) vs Analysis": "aurora_oper_vs_hres_t0_1440x721_2022.nc",
        # "FGN (oper.) (1st member) vs Analysis": "fgn_oper_first_member_vs_hres_t0_1440x721_2022.nc",
        # "FGN (oper.) (1st member) vs ERA5": "fgn_oper_first_member_vs_era5_1440x721_2022.nc",
        "FGN (oper.) (mean) vs Analysis": "fgn_oper_mean_vs_hres_t0_1440x721_2022.nc",
        "FGN (oper.) (mean) vs ERA5": "fgn_oper_mean_vs_era5_1440x721_2022.nc",
        # "GenCast (oper.) (1st member) vs Analysis": "gencast_oper_first_member_vs_hres_t0_1440x721_2022.nc",
        # "GenCast (oper.) (1st member) vs ERA5": "gencast_oper_first_member_vs_era5_1440x721_2022.nc",
        "GenCast (oper.) (mean) vs Analysis": "gencast_oper_mean_vs_hres_t0_1440x721_2022.nc",
        "GenCast (oper.) (mean) vs ERA5": "gencast_oper_mean_vs_era5_1440x721_2022.nc",
        "GraphCast (oper.) vs Analysis": "graphcast_oper_vs_hres_t0_1440x721_2022.nc",
        "GraphCast (oper.) vs ERA5": "graphcast_oper_vs_era5_1440x721_2022.nc",
        "GraphCast vs ERA5": "graphcast_vs_era5_1440x721_2022.nc",
        # "IFS ENS (1st member) vs Analysis": "ifs_ens_first_member_vs_hres_t0_1440x721_2022.nc",
        # "IFS ENS (1st member) vs ERA5": "ifs_ens_first_member_vs_era5_1440x721_2022.nc",
        "IFS ENS (mean) vs Analysis": "ifs_ens_mean_vs_hres_t0_1440x721_2022.nc",
        "IFS ENS (mean) vs ERA5": "ifs_ens_mean_vs_era5_1440x721_2022.nc",
        "IFS HRES vs Analysis": "ifs_hres_vs_hres_t0_1440x721_2022.nc",
        "IFS HRES vs ERA5": "ifs_hres_vs_era5_1440x721_2022.nc",
        # "Pangu-Weather (oper.) vs Analysis": "pangu_weather_oper_vs_hres_t0_1440x721_2022.nc",
        # "Pangu-Weather (oper.) vs ERA5": "pangu_weather_oper_vs_era5_1440x721_2022.nc",
        # "Pangu-Weather vs ERA5": "pangu_weather_vs_era5_1440x721_2022.nc",
    }
}[mode]

SURFACE_VARIABLES = ['2m_temperature', 'sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind']
ATMOSPHERIC_VARIABLES = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity']

PLOTS_DIR = "scraping/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*', 'h', '8', 'p']

def clean_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "")
        .replace("-", "_")
    )


def build_model_styles(model_labels):
    colors = list(plt.get_cmap("tab20").colors)
    n_colors = len(colors)
    styles = {}
    for idx, label in enumerate(model_labels):
        styles[label] = {
            "color": colors[idx % n_colors],
            "marker": MARKERS[(idx // n_colors) % len(MARKERS)],
            "linestyle": "-",
            "linewidth": 2.0,
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgewidth": 1.2,
            "markevery": 2,
        }
    return styles

def plot_metric(metric, datasets, labels, which_variables, sel_kwargs, logy=False, unitless=False, title=None, model_styles=None):
    if which_variables == "surface":
        variables = SURFACE_VARIABLES
    elif which_variables == "atmospheric":
        variables = ATMOSPHERIC_VARIABLES
    
    datasets = [ds.sel(**sel_kwargs, drop=True) for ds in datasets]
    
    nvars = len(variables)
    levels = datasets[0].level.values if which_variables == "atmospheric" else [None]
    nlevels = datasets[0].sizes["level"] if which_variables == "atmospheric" else 1
    
    lead_times = datasets[0]["lead_time"].values.astype("timedelta64[h]").astype(int)
    
    fig, axs = plt.subplots(ncols=nvars, nrows=nlevels, figsize=(5*nvars, 5*nlevels))
    if nlevels == 1:
        axs = np.expand_dims(axs, axis=0)
        sel_level_fn = lambda ds, j: ds
    else:
        sel_level_fn = lambda ds, j: ds.sel(level=ds.level.values[j])
    
    for i, var in enumerate(variables):
        for j in range(nlevels):
            varname = var if which_variables == "surface" else f"{var}_{levels[j]}"
            for ds, label in zip(datasets, labels):
                style_kwargs = model_styles.get(label, {}) if model_styles is not None else {}
                axs[j,i].plot(
                    lead_times,
                    sel_level_fn(ds, j)[f"{metric}.{var}"],
                    label=label,
                    **style_kwargs,
                )
            axs[j,i].set_xlabel("Lead time (hours)")
            if logy:
                axs[j,i].set_yscale("log")
            if j == 0:
                axs[j,i].set_title('')
            if unitless:
                axs[j,i].set_ylabel(f"{varname}")
            else:
                axs[j,i].set_ylabel(f"{varname}")
    
    if title is not None:
        fig.suptitle(title)
    else:
        metric_name = [part.capitalize() for part in metric.split("_")]
        metric_name[-1] = metric_name[-1].upper()
        fig.suptitle(" ".join(metric_name))
                
    offset = -0.2 if which_variables == 'surface' else -0.03
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, offset), ncol=5)

    bottom_margin = 0.12 if which_variables == 'surface' else 0.08
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.96])
    return fig
    
if __name__ == "__main__":
    # load all datasets
    datasets = {k: xr.open_dataset(os.path.join(OUTPUT_DIR, v)) for k, v in FILENAMES.items()}
    model_styles = build_model_styles(list(FILENAMES.keys()))
    
    # plot each metric
    for metric in METRICS:
        for which_variables in ["surface", "atmospheric"]:
            if metric == "SEEPS":
                continue
            fig = plot_metric(
                metric=clean_name(metric),
                datasets=[v for k, v in datasets.items()],
                labels=[k for k in datasets.keys()],
                which_variables=which_variables,
                sel_kwargs={"region": "Global"},
                logy=metric in ["MSE", "CRPS"],
                title=f"{metric} - Global",
                model_styles=model_styles,
            )
            fig.savefig(
                os.path.join(PLOTS_DIR, f"{mode}_{clean_name(metric)}_{which_variables}.png"),
                bbox_inches="tight"
            )