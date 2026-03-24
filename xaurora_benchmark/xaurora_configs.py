# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private configs"""

import copy

years = [2021, 2022]

upper_level_variables = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    # 'wind_speed',
    'specific_humidity',
]

surface_variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    # '10m_wind_speed',
]

standard_variables = upper_level_variables + surface_variables

gc_kwargs = {
    'rename_dimensions': {
        'time': 'init_time',
        'prediction_timedelta': 'lead_time',
        'lat': 'latitude',
        'lon': 'longitude',
    }
}
deterministic_prediction_configs = {
    # ERA5
    **dict.fromkeys(
        [f'era5_1440x721_{y}' for y in years],
        {
            'path': '/gpfs/work3/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2/',
            'variables': standard_variables,
        },
    ),
    # HRES
    **dict.fromkeys(
        [f'hres_1440x721_{y}' for y in years],
        {
            'path': 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr',
            'variables': standard_variables,
        },
    ),
    **dict.fromkeys(
        [f'hres_64x32_{y}' for y in years],
        {
            'path': 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr',
            'variables': standard_variables,
        },
    ),
    # ENS (Mean)
    **dict.fromkeys(
        [
            f'ens_mean_1440x721_{y}' for y in years
        ],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr',
            'variables': standard_variables,
            'levels': [500, 700, 850],
        },
    ),
    ### ****************************************************
    ###        PRE-PATCH EXPERIMENTS (DEPRECATED)
    ### ****************************************************
    # # Aurora Small vs ERA5 (15d/2021)
    # 'aurora_small_pretrained_1440x721_2021': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/small-flow-map_600K_0.1-noise/forecast_20-SDE-steps/aurora-small-pretrained_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Aurora PT vs ERA5 (15d/2021)
    # 'aurora_pretrained_1440x721_2021': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2021_15d_20-SDE-steps/aurora-pretrained_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Aurora PT vs ERA5 (15d/2022)
    # 'aurora_pretrained_1440x721_2022': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2022_15d_20-SDE-steps/aurora-pretrained_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Aurora FT (no LoRA) vs HRES-t0 (15d/2022)
    # 'aurora_finetuned_no_lora_hres_init_1440x721_2022': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_hres_init_2022_15d_20-SDE-steps/aurora-finetuned_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Aurora FT (no LoRA) vs HRES-t0 (15d/2022)
    # 'aurora_finetuned_no_lora_hres_init_1440x721_2022': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_hres_init_2022_15d_20-SDE-steps/aurora-finetuned_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Aurora FT (oper.) vs HRES-t0 (15d/2022)
    # 'aurora_finetuned_hres_init_1440x721_2022': {
    #     'path': '/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise_ft-hres-nowd/forecast_hres_init_2022_15d_20-SDE-steps/aurora-finetuned_forecasts.zarr',
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # # Aurora FT (oper.) (from aurora code) vs HRES-t0 (15d/2022)
    # 'aurora_finetuned_from_aurora_hres_init_1440x721_2022': {
    #     "path": "/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_hres_init_2022_15d/aurora-finetuned-from-aurora_forecasts.zarr",
    #     "variables": standard_variables,
    #     "levels": [250, 500, 700, 850]
    # },
    
    ### ****************************************************
    ###        POST-PATCH EXPERIMENTS (VALIDATED)
    ### ****************************************************
    # Aurora Small vs ERA5 (5d/2022)
    'aurora_small_pretrained_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-14_11-29-43/aurora-small-pretrained_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-14_11-29-43/aurora_small_pretrained_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Aurora PT vs ERA5 (15d/2022)
    'aurora_pretrained_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_xaurora/forecast_2026-03-10_20-46-40/aurora-pretrained_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_xaurora/forecast_2026-03-10_20-46-40/aurora_pretrained_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Aurora FT (no LoRA) vs HRES-t0 (15d/2022)
    'aurora_finetuned_no_lora_hres_init_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_14-13-58/aurora-finetuned-no-lora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_14-13-58/aurora_finetuned_no_lora_hres_init_vs_hres_t0_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Aurora FT (oper.) vs HRES-t0 (15d/2022)
    'aurora_finetuned_hres_init_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_13-56-18/aurora-finetuned_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_13-56-18/aurora_finetuned_hres_init_vs_hres_t0_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Aurora FT (oper.) (from aurora code) vs HRES-t0 (15d/2022)
    'aurora_finetuned_from_aurora_hres_init_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_11-28-34/aurora-finetuned-from-aurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/aurora_from_aurora/forecast_2026-03-11_11-28-34/aurora_finetuned_from_aurora_hres_init_vs_hres_t0_1440x721_2022-01-01_2022-12-31.nc',
        "variables": standard_variables,
        "levels": [250, 500, 700, 850]
    },
}
# For ensembles, add single member config
add_single_member_config = ['ens', 'neuralgcm_ens']


def select_first_member(ds):
  if 'number' in ds.dims:
    return ds.isel(number=0)
  elif 'sample' in ds.dims:
    return ds.isel(sample=0)
  elif 'member' in ds.dims:
    return ds.isel(member=0)
  elif 'realization' in ds.dims:
    return ds.isel(realization=0)
  else:
    raise ValueError('Dataset does not have a member dimension.')


single_member_configs = {}
for model, config in deterministic_prediction_configs.items():
  if any(model.startswith(m) for m in add_single_member_config):
    single_member_config = copy.deepcopy(config)
    single_member_config['path'] = single_member_config['path'].replace(  # pytype: disable=attribute-error
        '_mean.zarr', '.zarr'
    )
    if 'data_loader_kwargs' in single_member_config:
      assert (
          'preprocessing_fn' not in single_member_config['data_loader_kwargs']
      )
      single_member_config['data_loader_kwargs'][
          'preprocessing_fn'
      ] = select_first_member
    else:
      single_member_config['data_loader_kwargs'] = {
          'preprocessing_fn': select_first_member
      }
    single_member_configs[model.replace('_mean', '_single_member')] = (
        single_member_config
    )
deterministic_prediction_configs.update(single_member_configs)

probabilistic_prediction_configs = {
    # IFS ENS
    **dict.fromkeys(
        [
            f'ens_1440x721_{y}' 
            for y in years
        ],
        {
            'path': (
                'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr'
            ),
            'variables': standard_variables,
            'levels': [500, 700, 850],
        },
    ),
    ### ****************************************************
    ###        PRE-PATCH EXPERIMENTS (DEPRECATED)
    ### ****************************************************
    # # Xaurora Small vs ERA5 (15d/2021)
    # 'xaurora_small_1440x721_2021': {
    #     'path': '/projects/prjs1808/ewalt1/Xaurora/train/small-flow-map_600K_0.1-noise/forecast_20-SDE-steps/xaurora_forecasts.zarr',
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # # Xaurora Small FM vs ERA5 (15d/2021) 
    # 'xaurora_small_fm_1440x721_2021': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/16g/2026-02-10_15-11-33/forecast_2026-02-16_13-05-34/xaurora_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # Xaurora vs ERA5 (15d/2021)
    # 'xaurora_1440x721_2021': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2021_15d_20-SDE-steps/xaurora_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # # Xaurora vs ERA5 (15d/2022)
    # 'xaurora_1440x721_2022': {
    #     # TODO: update #  'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2022_15d_20-SDE-steps/xaurora_forecasts.zarr",
    #     'path': '/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2026-03-11_18-15-50/xaurora_forecasts.zarr',
    #     'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2026-03-11_18-15-50/xaurora_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # # Xaurora FT vs HRES-t0 (zero-shot) (15d/2022)
    # 'xaurora_zero_shot_hres_init_1440x721_2022': {
    #     'path': "/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_hres_init_2022_15d_20-SDE-steps/xaurora_forecasts.zarr",
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    # # Xaurora FT vs HRES-t0 (15d/2022)
    # 'xaurora_finetuned_hres_init_1440x721_2022': {
    #     'path': '/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise_ft-hres-nowd/forecast_hres_init_2022_15d_20-SDE-steps/xaurora_forecasts.zarr',
    #     'variables': standard_variables,
    #     'levels': [250, 500, 700, 850]
    # },
    
    ### ****************************************************
    ###                  SI ABLATIONS
    ### ****************************************************
    # Xaurora-Small-FlowMapSI vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-14_11-29-43/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-14_11-29-43/xaurora_small_flowmapsi_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-FollmerSI vs ERA5 (5d/2022)
    'xaurora_small_follmersi_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-15/forecast_2026-03-14_13-28-24/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-15/forecast_2026-03-14_13-28-24/xaurora_small_follmersi_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    
    ### ****************************************************
    ###               GENERATIVE ABLATIONS
    ### ****************************************************
    # Xaurora-Small-FlowMatching vs ERA5 (5d/2022)
    'xaurora_small_flowmatching_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_18-46-26/forecast_2026-03-16_23-25-31/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_18-46-26/forecast_2026-03-16_23-25-31/xaurora_small_flowmatching_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-DDPM vs ERA5 (5d/2022)
    'xaurora_small_ddpm_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_18-46-26/forecast_2026-03-17_09-27-37/xaurora_forecasts_ddpm.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_18-46-26/forecast_2026-03-17_09-27-37/xaurora_small_ddpm_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    
    ### ****************************************************
    ###                 NOISE ABLATIONS
    ### ****************************************************
    # Xaurora-Small-FlowMapSI-0.25 vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_025noise_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_14-41-31/forecast_2026-03-16_23-25-31/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_14-41-31/forecast_2026-03-16_23-25-31/xaurora_small_flowmapsi_025noise_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-FlowMapSI-0.5 vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_05noise_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_15-14-32/forecast_2026-03-16_23-25-31/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-15_15-14-32/forecast_2026-03-16_23-25-31/xaurora_small_flowmapsi_05noise_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-FlowMapSI-1.0 vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_1.0noise_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-23_17-32-48/forecast_2026-03-24_16-00-54/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-23_17-32-48/forecast_2026-03-24_16-00-54/xaurora_small_flowmapsi_10noise_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },    
    # Xaurora-Small-FlowMapSI-infer0.25 vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_infer0.25_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-17_12-06-14/xaurora_forecasts_infer0.25.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-17_12-06-14/xaurora_small_flowmapsi_infer0.25_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]    
    },
    # Xaurora-Small-FlowMapSI-infer0.5 vs ERA5 (5d/2022)
    'xaurora_small_flowmapsi_infer0.5_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-17_12-06-54/xaurora_forecasts_infer0.5.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-13_11-18-14/forecast_2026-03-17_12-06-54/xaurora_small_flowmapsi_infer0.5_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-FlowMapSI-gamma3-v0_0-v1_-3
    'xaurora_small-flowmapsi_gamma3_v00_v1-3_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-18_12-24-35/forecast_2026-03-23_17-25-28/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-18_12-24-35/forecast_2026-03-23_17-25-28/xaurora_small-flowmapsi_gamma3_v00_v1-3_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
    # Xaurora-Small-FlowMapSI-gamma3-v0_3-v1_0
    'xaurora_small-flowmapsi_gamma3_v03_v10_1440x721_2022': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-18_12-24-41/forecast_2026-03-23_17-31-18/xaurora_forecasts.zarr',
        'metrics_path': '/projects/prjs1808/ewalt1/Xaurora/train/8g/2026-03-18_12-24-41/forecast_2026-03-23_17-31-18/xaurora_small-flowmapsi_gamma3_v03_v10_vs_era5_1440x721_2022-01-01_2022-12-31.nc',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
}

target_configs = {
    # ERA5
    'era5_1440x721': {
        'path': '/gpfs/work3/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2/',
        'path_2022': "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr/",
        'variables': standard_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    'era5_64x32': {
        'path': 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    # HRES T0
    'hres_t0_1440x721': {
        'path': '/projects/prjs1808/datasets/HRES_T0/datasets/hres_t0/2016-2022-6h-1440x721.zarr/',
        'variables': standard_variables,
    },
    'hres_t0_64x32': {
        'path': 'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
}

climatology_configs = {
    **dict.fromkeys(
        [f'era5_1440x721_{y}' for y in years],
        {
            'path': '/gpfs/work2/0/prjs0981/datasets/ERA5/era5-zarr/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr/',
            'variables': standard_variables,
            'data_loader_kwargs': {
                'preprocessing_fn': lambda ds: ds.sortby('latitude')
            },
        },
    ),
    **dict.fromkeys(
        [f'era5_64x32_{y}' for y in years],
        {
            'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr',
            'variables': standard_variables,
        },
    ),
}
