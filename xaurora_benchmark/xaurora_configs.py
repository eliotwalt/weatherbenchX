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

years = [2021]

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
    # HRES T0
    **dict.fromkeys(
        [f'hres_t0_1440x721_{y}' for y in years],
        {
            'path': '/projects/prjs1808/datasets/HRES_T0/datasets/hres_t0/2016-2022-6h-1440x721.zarr/',
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
    # Aurora small pretrained
    'aurora_small_pretrained_1440x721_2021': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/16g/2026-01-23_10-18-59/forecast_2026-02-11_15-55-40/aurora-small-pretrained_forecasts.zarr',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
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
    # Xaurora
    'xaurora_small_1440x721_2021': {
        'path': '/projects/prjs1808/ewalt1/Xaurora/train/16g/2026-01-23_10-18-59/forecast_2026-02-11_15-55-40/xaurora_forecasts.zarr',
        'variables': standard_variables,
        'levels': [250, 500, 700, 850]
    },
}

target_configs = {
    # ERA5
    'era5_1440x721': {
        'path': '/gpfs/work3/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2/',
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
}
