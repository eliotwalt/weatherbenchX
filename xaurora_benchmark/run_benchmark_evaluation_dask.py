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
r"""Evaluation script using pure xarray/dask (no Apache Beam).

Example usage:

python run_benchmark_evaluation_dask.py \
  --config=xaurora_configs \
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
  --n_workers=4
"""
from collections.abc import Sequence
import importlib
import os
from typing import Mapping

from absl import app
from absl import flags
from absl import logging
import dask
from dask.diagnostics import ProgressBar
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import binning
from weatherbenchX import weighting
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import categorical
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import probabilistic
from weatherbenchX.metrics import wrappers
import xarray as xr


CONFIG = flags.DEFINE_string('config', None, 'Config module name.')
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
    'lead_time_stop', None, help='Lead time end in hours (exclusive).'
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
N_WORKERS = flags.DEFINE_integer(
    'n_workers', 1, 'Number of dask workers for parallel processing.'
)


_DEFAULT_LEVELS = [500, 700, 850]

REGIONS = {
    # ECMWF regions
    'global': ((-90, 90), (0, 360)),
    'tropics': ((-20, 20), (0, 360)),
    'northern-hemisphere': ((20, 90), (0, 360)),
    'southern-hemisphere': ((-90, -20), (0, 360)),
    'europe': ((35, 75), (-12.5, 42.5)),
    'mediterranean': ((25, 50), (-10, 40)),
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


def process_chunk(
    init_times: np.ndarray,
    lead_times: np.ndarray,
    prediction_loader,
    target_loader,
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
) -> aggregation.AggregationState:
  """Process a single time chunk and return aggregated statistics.

  Args:
    init_times: Array of initialization times for this chunk.
    lead_times: Array of lead times for this chunk.
    prediction_loader: DataLoader for predictions.
    target_loader: DataLoader for targets.
    metrics: Dictionary of metrics to compute.
    aggregator: Aggregator instance.

  Returns:
    AggregationState containing aggregated statistics for this chunk.
  """
  # Load targets and predictions for this chunk
  targets_chunk = target_loader.load_chunk(init_times, lead_times)
  predictions_chunk = prediction_loader.load_chunk(
      init_times, lead_times, targets_chunk
  )

  # Compute statistics
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )

  # Aggregate statistics
  aggregation_state = aggregator.aggregate_statistics(statistics)

  return aggregation_state


def generate_time_chunks(
    init_times: np.ndarray,
    lead_times: np.ndarray,
    init_time_chunk_size: int,
    lead_time_chunk_size: int,
):
  """Generate chunks of init_times and lead_times.

  Args:
    init_times: Full array of initialization times.
    lead_times: Full array of lead times.
    init_time_chunk_size: Number of init times per chunk.
    lead_time_chunk_size: Number of lead times per chunk.

  Yields:
    Tuples of (init_times_chunk, lead_times_chunk).
  """
  n_init_chunks = int(np.ceil(len(init_times) / init_time_chunk_size))
  n_lead_chunks = int(np.ceil(len(lead_times) / lead_time_chunk_size))

  for i in range(n_init_chunks):
    init_start = i * init_time_chunk_size
    init_end = min((i + 1) * init_time_chunk_size, len(init_times))
    init_chunk = init_times[init_start:init_end]

    for j in range(n_lead_chunks):
      lead_start = j * lead_time_chunk_size
      lead_end = min((j + 1) * lead_time_chunk_size, len(lead_times))
      lead_chunk = lead_times[lead_start:lead_end]

      yield init_chunk, lead_chunk


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

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
  levels = prediction_config.get('levels', _DEFAULT_LEVELS)
  prediction_loader_kwargs = prediction_config.get('data_loader_kwargs', {})

  # Create prediction loader
  if PREDICTION.value == 'persistence':
    prediction_loader_cls = xarray_loaders.PersistenceFromXarray
  elif PREDICTION.value == 'climatology':
    prediction_loader_cls = xarray_loaders.ClimatologyFromXarray
  elif PREDICTION.value == 'probabilistic_climatology':
    prediction_loader_cls = xarray_loaders.ProbabilisticClimatologyFromXarray
    prediction_loader_kwargs['start_year'] = 1990
    prediction_loader_kwargs['end_year'] = 2019
  else:
    prediction_loader_cls = xarray_loaders.PredictionsFromXarray

  prediction_loader = prediction_loader_cls(
      path=prediction_config['path'],
      variables=variables,
      sel_kwargs={'level': levels},
      **prediction_loader_kwargs,
  )
  prediction_loader.maybe_prepare_dataset()

  target_loader = xarray_loaders.TargetsFromXarray(
      path=target_config['path'],
      variables=variables,
      sel_kwargs={'level': levels},
      preprocessing_fn=lambda ds: ds.sortby('latitude'),
  )
  target_loader.maybe_prepare_dataset()

  # Reindex prediction to fit targets
  prediction_loader._ds = prediction_loader._ds.reindex(
      latitude=target_loader._ds['latitude'],
      method='nearest',
      tolerance=0.25,
  )

  ##############################################################################
  # 2. Set up time ranges
  ##############################################################################
  if (INIT_TIME_START.value is None) != (INIT_TIME_STOP.value is None):
    raise ValueError(
        'Init time start and stop must be both specified or both None.'
    )

  if INIT_TIME_START.value is None:
    init_time_start = f'{YEAR.value}-01-01T00'
    if YEAR.value == '2022' or PREDICTION.value in ['fuxi', 'excarta']:
      init_time_stop = f'{YEAR.value}-12-16T00'
      if PREDICTION.value == 'aurora':
        init_time_start = f'{YEAR.value}-01-01T12'
    elif YEAR.value == '2020' and PREDICTION.value == 'baguan':
      init_time_stop = f'{YEAR.value}-12-30T12'
    else:
      init_time_stop = f'{int(YEAR.value) + 1}-01-01T00'
    init_time_str = str(YEAR.value)
  else:
    init_time_start = INIT_TIME_START.value
    init_time_stop = INIT_TIME_STOP.value
    init_time_str = f'{init_time_start}_{init_time_stop}'

  if INIT_TIME_FREQUENCY.value is None:
    init_time_frequency = np.timedelta64(
        24 if PREDICTION.value == 'excarta' else 12, 'h'
    )
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
    if PREDICTION.value in ['persistence', 'climatology', 'probabilistic_climatology']:
      lead_times = np.arange(0, 15 * 24 + 6, 6, dtype='timedelta64[h]')
    else:
      lead_times = prediction_loader._ds.lead_time.values
  else:
    logging.info(
        f'Creating lead times from {LEAD_TIME_START.value} to '
        f'{LEAD_TIME_STOP.value} every {LEAD_TIME_FREQUENCY.value} hours.'
    )
    lead_times = np.arange(
        LEAD_TIME_START.value,
        LEAD_TIME_STOP.value,
        LEAD_TIME_FREQUENCY.value,
        dtype='timedelta64[h]',
    )

  # Intersect with available times in prediction dataset
  pred_time_dim = 'time' if 'time' in prediction_loader._ds.dims else 'init_time'
  init_times = np.intersect1d(
      init_times,
      prediction_loader._ds[pred_time_dim].values,
  )
  assert len(init_times) > 0, 'No init times available for evaluation.'

  pred_lead_dim = (
      'lead_time' if 'lead_time' in prediction_loader._ds.dims
      else 'prediction_timedelta'
  )
  available_lead_times = (
      prediction_loader._ds[pred_lead_dim].values
      .astype('timedelta64[h]')
      .astype(int)
  )
  lead_times = np.intersect1d(lead_times, available_lead_times)
  assert len(lead_times) > 0, 'No lead times available for evaluation.'

  logging.info(
      f'Number of init times: {len(init_times)}, '
      f'first: {init_times[0]}, last: {init_times[-1]}'
  )
  logging.info(f'Lead times: {lead_times}')

  ##############################################################################
  # 3. Define metrics
  ##############################################################################
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

  u_names, v_names, vector_names = [], [], []
  if 'u_component_of_wind' in variables and 'v_component_of_wind' in variables:
    u_names.append('u_component_of_wind')
    v_names.append('v_component_of_wind')
    vector_names.append('wind')
  if ('10m_u_component_of_wind' in variables and
      '10m_v_component_of_wind' in variables):
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
          [wrappers.EnsembleMean(which='predictions')],
      ),
  }

  all_metrics = probabilistic_metrics if is_probabilistic else deterministic_metrics

  ##############################################################################
  # 4. Define aggregation method
  ##############################################################################
  land_sea_mask = xr.open_zarr(
      configs.target_configs[f'era5_{RESOLUTION.value}']['path']
  )['land_sea_mask'].compute()
  land_sea_mask = land_sea_mask.sel(
      latitude=prediction_loader._ds.latitude,
      longitude=prediction_loader._ds.longitude,
  )

  bin_by = [binning.Regions(REGIONS, land_sea_mask=land_sea_mask)]

  reduce_dims = (
      ['latitude', 'longitude'] if TEMPORAL.value
      else ['init_time', 'latitude', 'longitude']
  )
  aggregator = aggregation.Aggregator(
      reduce_dims=reduce_dims,
      weigh_by=[weighting.GridAreaWeighting()],
      bin_by=bin_by,
      masked=True,
      skipna=True if PREDICTION.value == 'keisler' else False,
  )

  ##############################################################################
  # 5. Process chunks and aggregate
  ##############################################################################
  filename = (
      f'{PREDICTION.value}_vs_{TARGET.value}_{RESOLUTION.value}_{init_time_str}'
  )
  if TEMPORAL.value:
    filename += '_temporal'
  filename += '.nc'
  out_path = os.path.join(OUTPUT_DIR.value, filename)
  logging.info(f'Save path: {out_path}')

  # Generate all chunks
  chunks = list(generate_time_chunks(
      init_times,
      lead_times,
      INIT_TIME_CHUNK_SIZE.value,
      LEAD_TIME_CHUNK_SIZE.value,
  ))
  logging.info(f'Processing {len(chunks)} chunks...')

  if N_WORKERS.value > 1:
    # Parallel processing with dask.delayed
    @dask.delayed
    def delayed_process_chunk(init_chunk, lead_chunk):
      return process_chunk(
          init_chunk,
          lead_chunk,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregator,
      )

    # Create delayed tasks for all chunks
    delayed_results = [
        delayed_process_chunk(init_chunk, lead_chunk)
        for init_chunk, lead_chunk in chunks
    ]

    # Compute all chunks in parallel
    logging.info(f'Computing {len(delayed_results)} chunks with {N_WORKERS.value} workers...')
    with ProgressBar():
      with dask.config.set(num_workers=N_WORKERS.value):
        aggregation_states = dask.compute(*delayed_results)
  else:
    # Sequential processing
    aggregation_states = []
    for i, (init_chunk, lead_chunk) in enumerate(chunks):
      logging.info(f'Processing chunk {i + 1}/{len(chunks)}...')
      agg_state = process_chunk(
          init_chunk,
          lead_chunk,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregator,
      )
      aggregation_states.append(agg_state)

  # Combine all aggregation states
  logging.info('Combining aggregation states...')
  final_state = aggregation.AggregationState.sum(aggregation_states)

  # Compute final metric values
  logging.info('Computing final metrics...')
  results = final_state.metric_values(all_metrics)

  # Remove attributes and save
  results = results.drop_attrs(deep=True)

  # Create output directory if needed
  os.makedirs(OUTPUT_DIR.value, exist_ok=True)

  logging.info(f'Saving results to {out_path}...')
  results.to_netcdf(out_path)
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)

