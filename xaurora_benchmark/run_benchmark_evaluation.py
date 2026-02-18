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
r"""Evaluation script for the public benchmark.

Example usage:

export BUCKET=my-bucket
export PROJECT=my-project
export REGION=us-central1

python run_benchmark_evaluation.py \
  --config=public_configs \
  --prediction=hres \
  --target=era5 \
  --resolution=64x32 \
  --year=2020 \
  --time_start=2020-01-01 \
  --time_stop=2020-01-01T12 \
  --lead_time_start=0 \
  --lead_time_stop=12 \
  --lead_time_frequency=6 \
  --output_dir=./results/ \
  --runner=DirectRunner

  or to run on DataFlow:
  --output_dir=gs://$BUCKET/tmp/ \
  --runner=DataflowRunner \
  -- \
  --project=$PROJECT \
  --region=$REGION \
  --temp_location=gs://$BUCKET/tmp/ \
  --setup_file=../setup.py \
  --job_name=wbx-evaluation
"""
from collections.abc import Sequence
import copy
import importlib
import os
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import beam_pipeline
from weatherbenchX import binning
from weatherbenchX import time_chunks
from weatherbenchX import weighting
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import categorical
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import probabilistic
from weatherbenchX.metrics import wrappers
import xarray as xr

from utils import apply_convolve_fill_nan

CONFIG = flags.DEFINE_string('config', None, 'beam.runners.Runner')
PREDICTION = flags.DEFINE_string('prediction', None, 'Prediction name.')
TARGET = flags.DEFINE_string('target', None, 'Target name.')
RESOLUTION = flags.DEFINE_string('resolution', None, 'Resolution, e.g. 240x121')
YEAR = flags.DEFINE_string('year', None, 'Year to evaluate.')
INIT_TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='Timestamp (inclusive) at which to start evaluation',
)
INIT_TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='Timestamp (exclusive) at which to stop evaluation',
)
INIT_TIME_FREQUENCY = flags.DEFINE_integer(
    'time_frequency', 6, help='Init frequency.'
)
LEAD_TIME_START = flags.DEFINE_integer(
    'lead_time_start', None, help='Lead time start in hours.'
)
LEAD_TIME_STOP = flags.DEFINE_integer(
    'lead_time_stop', None, help='Lead time end in hours(exclusive).'
)
LEAD_TIME_FREQUENCY = flags.DEFINE_integer(
    'lead_time_frequency', 6, help='Lead time frequency in hours.'
)
OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to save results.'
)
INIT_TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'init_time_chunk_size', 1, 'Init time chunk size.'
)
LEAD_TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'lead_time_chunk_size', 12, 'Lead time chunk size.'
)
TEMPORAL = flags.DEFINE_bool(
    'temporal', False, 'If true, do not reduce over init time.'
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


_DEFAULT_LEVELS = [500, 700, 850]

REGIONS = {
    # ECMWF regions
    'global': ((-90, 90), (0, 360)),
    'tropics': ((-20, 20), (0, 360)),
    # TODO(srasp): Add extra-tropics.
    'northern-hemisphere': ((20, 90), (0, 360)),
    'southern-hemisphere': ((-90, -20), (0, 360)),
    'europe': ((35, 75), (-12.5, 42.5)),
    'mediterranean': ((25, 50), (-10, 40)), # mine
    'north-america': ((25, 60), (360 - 120, 360 - 75)),
    'north-atlantic': ((25, 65), (360 - 70, 360 - 10)),
    'north-pacific': ((25, 60), (145, 360 - 130)),
    'east-asia': ((25, 60), (102.5, 150)),
    'ausnz': ((-45, -12.5), (120, 175)),
    'arctic': ((60, 90), (0, 360)),
    'antarctic': ((-90, -60), (0, 360)),
    # Additional regions
    'northern-africa': ((5, 32.5), (-12.5, 37.5)),
    'southern-africa': ((-30, 5), (12.5, 37.5)),
    'south-america': ((-40, 5), (-75, -45)),
    'west-asia': ((15, 60), (42.5, 102.5)),
    'south-east-asia': ((-12.5, 25), (95, 125)),
}


def main(argv: Sequence[str]) -> None:
  configs = importlib.import_module(CONFIG.value)

  ##############################################################################
  # 1. Get data loaders
  ##############################################################################
  is_probabilistic = False
  prediction_str_name = f'{PREDICTION.value}_{RESOLUTION.value}_{YEAR.value}'
  if PREDICTION.value == 'persistence':
    prediction_config = configs.target_configs[f'era5_{RESOLUTION.value}']
  elif PREDICTION.value == 'probabilistic_climatology':
    prediction_config = configs.target_configs[f'era5_{RESOLUTION.value}']
    is_probabilistic = True
  elif PREDICTION.value == 'climatology':
    prediction_config = configs.climatology_configs[
        f'era5_{RESOLUTION.value}_{YEAR.value}'
    ]
  elif prediction_str_name in configs.deterministic_prediction_configs:
    prediction_config = configs.deterministic_prediction_configs[
        prediction_str_name
    ]
  else:
    prediction_config = configs.probabilistic_prediction_configs[
        prediction_str_name
    ]
    is_probabilistic = True
  target_config = configs.target_configs[f'{TARGET.value}_{RESOLUTION.value}']
  climatology_config = configs.climatology_configs[
      f'era5_{RESOLUTION.value}_{YEAR.value}'
  ]
  variables = np.intersect1d(
      prediction_config['variables'], target_config['variables']
  )
  precip_variables = [
      v for v in variables if v.startswith('total_precipitation')
  ]
  levels = (
      prediction_config['levels']
      if 'levels' in prediction_config
      else _DEFAULT_LEVELS
  )
  prediction_loader_kwargs = (
      prediction_config['data_loader_kwargs']
      if 'data_loader_kwargs' in prediction_config
      else {}
  )
  if PREDICTION.value == 'persistence':
    prediction_loader = xarray_loaders.PersistenceFromXarray
  elif PREDICTION.value == 'climatology':
    prediction_loader = xarray_loaders.ClimatologyFromXarray
  elif PREDICTION.value == 'probabilistic_climatology':
    prediction_loader = xarray_loaders.ProbabilisticClimatologyFromXarray
    prediction_loader_kwargs['start_year'] = (
        1990  #   pytype: disable=unsupported-operands
    )
    prediction_loader_kwargs['end_year'] = (
        2019  #   pytype: disable=unsupported-operands
    )
  else:
    prediction_loader = xarray_loaders.PredictionsFromXarray
    
  # Deal with NaNs in HRES and HRES-T0
  if PREDICTION.value in ['hres', 'hres_t0']:
    prediction_process_chunk_fn = apply_convolve_fill_nan
  else: prediction_process_chunk_fn = None
  
  if TARGET.value == 'hres_t0':
    target_process_chunk_fn = apply_convolve_fill_nan
  else: target_process_chunk_fn = None
    
    
    
  prediction_loader = prediction_loader(
      path=prediction_config['path'],
      variables=variables,
      sel_kwargs={'level': levels},
      process_chunk_fn=prediction_process_chunk_fn,
      **prediction_loader_kwargs,
  )
  prediction_loader.maybe_prepare_dataset() # load such that we can access init times later
  target_loader = xarray_loaders.TargetsFromXarray(
      path=target_config['path'],
      variables=variables,
      sel_kwargs={'level': levels},
      process_chunk_fn=target_process_chunk_fn,
      # For some datasets, latitude is reversed. This isn't a problem per se,
      # as xarray alignes the datasets but we will still align here.
      # (It is a problem for the climatology, see below.)
      preprocessing_fn=lambda ds: ds.sortby('latitude'),
  )
  target_loader.maybe_prepare_dataset()
  
  # reindex prediction to fit targets
  prediction_loader._ds = prediction_loader._ds.reindex(latitude=target_loader._ds['latitude'], method="nearest", tolerance=0.25)
  
  # # Calculate the lat/lon bounds of the predictions
  # latitudes = prediction_loader._ds['latitude'].values
  # longitudes = prediction_loader._ds['longitude'].values
  
  # # select in target
  # target_loader._ds = target_loader._ds.sel(latitude=latitudes, longitude=longitudes)
  
  # # redefine regions
  # global REGIONS
  # lat_min, lat_max = latitudes.min().item(), latitudes.max().item()
  # lon_min, lon_max = longitudes.min().item(), longitudes.max().item()
  # REGIONS = {
  #     key: ((max(lat0, lat_min), min(lat1, lat_max)),
  #             (max(lon0, lon_min), min(lon1, lon_max)))
  #     for key, ((lat0, lat1),(lon0, lon1)) in REGIONS.items()
  # }
  # print(f"Using regions with lat/lon bounds: {REGIONS}")

  ##############################################################################
  # 2. Define time iterator
  ##############################################################################
  if (INIT_TIME_START.value is None) != (INIT_TIME_STOP.value is None):
    raise ValueError(
        'Init time start and stop must be both specified or both None.'
    )
  if INIT_TIME_START.value is None:
    init_time_start = f'{YEAR.value}-01-01T00'
    # Temporary hack for 2022 because target data only goes right until the end
    # of the year. Account for a 15 day max lead time.
    # Additionally, FuXi for 2020 only goes until mid-December.
    if YEAR.value == '2022' or PREDICTION.value in ['fuxi', 'excarta']:
      init_time_stop = f'{YEAR.value}-12-16T00'
      # First and last init are missing for aurora
      if PREDICTION.value == 'aurora':
        init_time_start = f'{YEAR.value}-01-01T12'
    elif YEAR.value == '2020' and PREDICTION.value == 'baguan':
      # Last day is missing for baguan
      init_time_stop = f'{YEAR.value}-12-30T12'
    else:
      init_time_stop = f'{int(YEAR.value) + 1}-01-01T00'
    init_time_str = str(YEAR.value)
  else:
    init_time_start = INIT_TIME_START.value
    init_time_stop = INIT_TIME_STOP.value
    init_time_str = f'{init_time_start}_{init_time_stop}'
  if INIT_TIME_FREQUENCY.value is None:
    if PREDICTION.value == 'excarta':
      init_time_frequency = np.timedelta64(24, 'h')
    else:
      init_time_frequency = np.timedelta64(12, 'h')
  else:
    init_time_frequency = np.timedelta64(INIT_TIME_FREQUENCY.value, 'h')
    
  init_times = np.arange(
      init_time_start,
      init_time_stop,
      init_time_frequency,
      dtype='datetime64',
  ).astype('datetime64[ns]')
  
  if (LEAD_TIME_START.value is None) != (LEAD_TIME_STOP.value is None):
    raise ValueError(
        'Lead time start and stop must be both specified or both None.'
    )
  if LEAD_TIME_START.value is None:
    logging.info('Using lead times from dataset.')
    if PREDICTION.value in [
        'persistence',
        'climatology',
        'probabilistic_climatology',
    ]:
      lead_times = np.arange(
          0,
          15 * 24 + 6,
          6,
          dtype='timedelta64[h]',
      )
    else:
      loader_copy = copy.copy(prediction_loader)
      loader_copy.maybe_prepare_dataset()
      assert loader_copy._ds is not None  # pylint: disable=protected-access
      lead_times = loader_copy._ds.lead_time.values # pylint: disable=protected-access
      
  else:
    logging.info(f"Creating lead times from {LEAD_TIME_START.value} to {LEAD_TIME_STOP.value} every {LEAD_TIME_FREQUENCY.value} hours.")
    lead_time_start = LEAD_TIME_START.value
    lead_time_stop = LEAD_TIME_STOP.value
    lead_times = np.arange(
        lead_time_start,
        lead_time_stop,
        LEAD_TIME_FREQUENCY.value,
        dtype='timedelta64[h]',
    )
    
  # intersect init_times with available init times in prediction dataset
  init_times_copy = copy.copy(init_times)
  init_times = np.intersect1d(
      init_times,
      prediction_loader._ds["time" if 'time' in prediction_loader._ds.dims else 'init_time'].values,
  )
  assert len(init_times) > 0, f'No init times available for evaluation. '\
    f'prediction init times: {prediction_loader._ds["time" if "time" in prediction_loader._ds.dims else "init_time"].values}, '\
    f'requested init times: {init_times_copy}'
  del init_times_copy
  
  # intersect lead_times with available lead times in prediction dataset
  lead_times_copy = copy.copy(lead_times)
  lead_times = np.intersect1d(
      lead_times,
      prediction_loader._ds["lead_time" if "lead_time" in prediction_loader._ds.dims else "prediction_timedelta"].values.astype('timedelta64[h]').astype(int),
  )
  assert len(lead_times) > 0, f'No lead times available for evaluation. '\
    f'prediction lead times: {prediction_loader._ds["lead_time" if "lead_time" in prediction_loader._ds.dims else "prediction_timedelta"].values}, '\
    f'requested lead times: {lead_times_copy}'
  del lead_times_copy
    
  times = time_chunks.TimeChunks(
      init_times,
      lead_times,
      init_time_chunk_size=INIT_TIME_CHUNK_SIZE.value,
      lead_time_chunk_size=LEAD_TIME_CHUNK_SIZE.value,
  )
  logging.info(f'Number of init times: {len(init_times)}, first init time: {init_times[0]}, last init time: {init_times[-1]}')
  logging.info(f'Lead times: {lead_times}')

  ##############################################################################
  # 3. Define metrics
  ##############################################################################
  # Here it is actually important to sort by latitude, because .where()
  # in the SEEPS computation isn't able to align the reversed latitude
  # coordinates.
  climatology = xr.open_zarr(climatology_config['path']).sortby('latitude')
  deterministic_metrics = {
      'rmse': deterministic.RMSE(),
      'mse': deterministic.MSE(),
      'bias': deterministic.Bias(),
      'acc': deterministic.ACC(climatology=climatology),
      'prediction_activity': deterministic.PredictionActivity(
          climatology=climatology
      ),
  }

  u_names = []
  v_names = []
  vector_names = []
  if 'u_component_of_wind' in variables and 'v_component_of_wind' in variables:
    u_names.append('u_component_of_wind')
    v_names.append('v_component_of_wind')
    vector_names.append('wind')
  if (
      '10m_u_component_of_wind' in variables
      and '10m_v_component_of_wind' in variables
  ):
    u_names.append('10m_u_component_of_wind')
    v_names.append('10m_v_component_of_wind')
    vector_names.append('10m_wind')
  if u_names:
    deterministic_metrics['vector_rmse'] = deterministic.WindVectorRMSE(
        u_names, v_names, vector_names
    )

  if precip_variables:
    seeps_dry_thresholds = {
        'total_precipitation_6hr': 0.1,
        'total_precipitation_24hr': 0.25,
    }
    deterministic_metrics['seeps'] = categorical.SEEPS(
        variables=precip_variables,
        climatology=climatology,
        dry_threshold_mm=[seeps_dry_thresholds[v] for v in precip_variables],
    )

  probabilistic_metrics = {
      'crps': probabilistic.CRPSEnsemble(),
      'unbiased_spread_skill': probabilistic.UnbiasedSpreadSkillRatio(),
      'unbiased_mean_rmse': probabilistic.UnbiasedEnsembleMeanRMSE(),
      'mean_rmse': wrappers.WrappedMetric(
          deterministic.RMSE(),
          [
              wrappers.EnsembleMean(
                  which='predictions',
              )
          ],
      ),
  }
  # TODO(srasp): Add brier scores for precipitation.

  if is_probabilistic:
    all_metrics = probabilistic_metrics
  else:
    all_metrics = deterministic_metrics

  ##############################################################################
  # 4. Define aggregation method
  ##############################################################################
  # Load land-sea mask from ERA5
  land_sea_mask = xr.open_zarr(
      configs.target_configs[f'era5_{RESOLUTION.value}']['path']
  )['land_sea_mask'].compute()
  # select only latitude/longitude points where we have predictions
  land_sea_mask = land_sea_mask.sel(
      latitude=prediction_loader._ds.latitude,
      longitude=prediction_loader._ds.longitude,
  )
  
  bin_by = [binning.Regions(REGIONS, land_sea_mask=land_sea_mask)]

  if TEMPORAL.value:
    reduce_dims = ['latitude', 'longitude']
  else:
    reduce_dims = ['init_time', 'latitude', 'longitude']
  aggregation_method = aggregation.Aggregator(
      reduce_dims=reduce_dims,
      weigh_by=[weighting.GridAreaWeighting()],
      bin_by=bin_by,
      masked=True,  # required for SEEPS and Keisler
      # Wind vector doesn't handle nan_mask properly at the moment.
      skipna=True if PREDICTION.value == 'keisler' else False,
  )

  ##############################################################################
  # 5. Define output path and run pipeline
  ##############################################################################
  filename = (
      f'{PREDICTION.value}_vs_{TARGET.value}_{RESOLUTION.value}_{init_time_str}'
  )
  if TEMPORAL.value:
    filename += '_temporal'
  filename += '.nc'
  out_path = os.path.join(OUTPUT_DIR.value, filename)
  logging.info(f'Save path: {out_path}')

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    beam_pipeline.define_pipeline(
        root,
        times,
        prediction_loader,
        target_loader,
        all_metrics,
        aggregation_method,
        out_path=out_path,
    )


if __name__ == '__main__':
  app.run(main)
