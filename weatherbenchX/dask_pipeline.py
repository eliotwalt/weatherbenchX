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
"""Defines a dask-based pipeline for evaluation.

This module provides a pure xarray/dask implementation of the evaluation
pipeline, serving as an alternative to the Apache Beam-based pipeline.
"""

import os
from typing import Callable, Mapping, Optional

from absl import logging
import dask
from dask.diagnostics import ProgressBar
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import time_chunks
from weatherbenchX.data_loaders import base as data_loaders_base
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


def _process_chunk(
    init_times: np.ndarray,
    lead_times: np.ndarray,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
) -> aggregation.AggregationState:
  """Process a single time chunk and return aggregated statistics.

  Args:
    init_times: Array of initialization times for this chunk.
    lead_times: Array of lead times for this chunk.
    predictions_loader: DataLoader for predictions.
    targets_loader: DataLoader for targets.
    metrics: Dictionary of metrics to compute.
    aggregator: Aggregator instance.

  Returns:
    AggregationState containing aggregated statistics for this chunk.
  """
  logging.log_first_n(
      logging.INFO,
      'Processing chunk: init_times=%s, lead_times=%s',
      10,
      init_times,
      lead_times,
  )

  # Load targets and predictions for this chunk
  targets_chunk = targets_loader.load_chunk(init_times, lead_times)
  predictions_chunk = predictions_loader.load_chunk(
      init_times, lead_times, targets_chunk
  )

  # Compute statistics
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )

  # Aggregate statistics
  aggregation_state = aggregator.aggregate_statistics(statistics)

  return aggregation_state


def run_pipeline(
    times: time_chunks.TimeChunks,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
    out_path: str | None = None,
    aggregation_state_out_path: str | None = None,
    setup_fn: Optional[Callable[[], None]] = None,
    n_workers: int = 1,
    show_progress: bool = True,
) -> xr.Dataset | None:
  """Runs the dask pipeline for calculating aggregated metrics.

  This function provides the same functionality as beam_pipeline.define_pipeline
  but uses dask for parallel processing instead of Apache Beam.

  Args:
    times: TimeChunks instance defining the chunks to process.
    predictions_loader: DataLoader instance for predictions.
    targets_loader: DataLoader instance for targets.
    metrics: A dictionary of metrics to compute.
    aggregator: Aggregation instance.
    out_path: The full path to write the metrics to (NetCDF format).
    aggregation_state_out_path: The full path to write the final aggregation
      state to. This can be useful if you want to compute further metrics from
      it later.
    setup_fn: (Optional) A function to call once before processing starts.
    n_workers: Number of dask workers for parallel processing.
    show_progress: Whether to show a progress bar during processing.

  Returns:
    The computed metrics as an xarray Dataset, or None if no out_path is
    specified.

  Raises:
    ValueError: If neither out_path nor aggregation_state_out_path is specified.
  """
  if out_path is None and aggregation_state_out_path is None:
    raise ValueError(
        'At least one of (metrics) out_path or aggregation_state_out_path must '
        'be specified.'
    )

  # Call setup function if provided
  if setup_fn is not None:
    setup_fn()

  # Generate all chunks as a list
  chunks = list(times)
  n_chunks = len(chunks)
  logging.info(f'Processing {n_chunks} chunks...')

  if n_workers > 1:
    # Parallel processing with dask.delayed
    @dask.delayed
    def delayed_process_chunk(init_chunk, lead_chunk):
      return _process_chunk(
          init_chunk,
          lead_chunk,
          predictions_loader,
          targets_loader,
          metrics,
          aggregator,
      )

    # Create delayed tasks for all chunks
    delayed_results = [
        delayed_process_chunk(init_chunk, lead_chunk)
        for init_chunk, lead_chunk in chunks
    ]

    # Compute all chunks in parallel
    logging.info(
        f'Computing {len(delayed_results)} chunks with {n_workers} workers...'
    )
    context_managers = []
    if show_progress:
      context_managers.append(ProgressBar())

    with dask.config.set(num_workers=n_workers):
      if show_progress:
        with ProgressBar():
          aggregation_states = dask.compute(*delayed_results)
      else:
        aggregation_states = dask.compute(*delayed_results)
  else:
    # Sequential processing
    aggregation_states = []
    for i, (init_chunk, lead_chunk) in enumerate(chunks):
      if show_progress:
        logging.info(f'Processing chunk {i + 1}/{n_chunks}...')
      agg_state = _process_chunk(
          init_chunk,
          lead_chunk,
          predictions_loader,
          targets_loader,
          metrics,
          aggregator,
      )
      aggregation_states.append(agg_state)

  # Combine all aggregation states
  logging.info('Combining aggregation states...')
  final_state = aggregation.AggregationState.sum(aggregation_states)

  # Write aggregation state if requested
  if aggregation_state_out_path is not None:
    logging.info(f'Writing aggregation state to {aggregation_state_out_path}...')
    aggregation_state_ds = final_state.to_dataset()
    # Remove attributes that may have been propagated from targets or predictions
    aggregation_state_ds = aggregation_state_ds.drop_attrs(deep=True)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(aggregation_state_out_path), exist_ok=True)
    aggregation_state_ds.to_netcdf(aggregation_state_out_path)

  # Compute and write final metric values if requested
  results = None
  if out_path is not None:
    logging.info('Computing final metrics...')
    results = final_state.metric_values(metrics)
    # Remove attributes that may have been propagated from targets or predictions
    results = results.drop_attrs(deep=True)
    # Ensure output directory exists
    output_dir = os.path.dirname(out_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Writing results to {out_path}...')
    results.to_netcdf(out_path)

  logging.info('Done!')
  return results


def run_unaggregated_pipeline(
    times: time_chunks.TimeChunks,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    out_path: str,
    zarr_chunks: Mapping[str, int] | None = None,
    setup_fn: Optional[Callable[[], None]] = None,
    n_workers: int = 1,
    show_progress: bool = True,
) -> None:
  """Runs a dask pipeline that calculates statistics without aggregation.

  Outputs statistics for all predictions and targets to a single Zarr store.
  This is equivalent to beam_pipeline.define_unaggregated_pipeline.

  Args:
    times: TimeChunks instance. Must have properly defined chunks.
    predictions_loader: DataLoader instance for predictions.
    targets_loader: DataLoader instance for targets.
    metrics: A dictionary of metrics to compute statistics for.
    out_path: The full path to write the output Zarr store to.
    zarr_chunks: (Optional) A dictionary of chunks to use for the output Zarr
      store. If None, the chunks will match those of TimeChunks.
    setup_fn: (Optional) A function to call once before processing starts.
    n_workers: Number of dask workers for parallel processing.
    show_progress: Whether to show a progress bar during processing.
  """
  # Call setup function if provided
  if setup_fn is not None:
    setup_fn()

  logging.info('Building template with data from first chunk...')

  # Get first chunk to build template
  first_chunk_index = 0
  try:
    first_init_times, first_lead_times = times[first_chunk_index]
  except IndexError:
    raise ValueError('Cannot generate template: TimeChunks is empty') from None

  targets_chunk = targets_loader.load_chunk(first_init_times, first_lead_times)
  predictions_chunk = predictions_loader.load_chunk(
      first_init_times, first_lead_times, targets_chunk
  )
  statistics_dict = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )

  # Build template dataset
  first_chunk_ds = xr.Dataset()
  for stat_name, var_dict in statistics_dict.items():
    for var_name, da in var_dict.items():
      first_chunk_ds[f'{stat_name}.{var_name}'] = da

  if 'mask' in first_chunk_ds.coords:
    raise ValueError(
        'mask coordinate found in template. add_nan_mask=True on data loaders '
        'is not supported for unaggregated pipelines.'
    )

  # Determine full dimensions
  template = first_chunk_ds.copy()

  if 'lead_time' in template.dims:
    vars_to_expand = [k for k, v in template.items() if 'lead_time' in v.dims]
    template = template.isel(lead_time=0, drop=True)
    lead_times = times.lead_times
    if isinstance(lead_times, slice):
      lead_times = np.arange(
          lead_times.start, lead_times.stop + lead_times.step, lead_times.step
      )
    for k in vars_to_expand:
      template[k] = template[k].expand_dims(lead_time=lead_times)

  if 'init_time' in template.dims:
    vars_to_expand = [k for k, v in template.items() if 'init_time' in v.dims]
    template = template.isel(init_time=0, drop=True)
    for k in vars_to_expand:
      template[k] = template[k].expand_dims(init_time=times.init_times)

  if 'init_time' in template.dims and 'lead_time' in template.dims:
    template.coords['valid_time'] = template.init_time + template.lead_time

  # Determine chunking
  dim_sizes = dict(template.sizes)
  stat_chunks = {}
  for dim, size in dim_sizes.items():
    if dim == 'init_time':
      stat_chunks[dim] = times.init_time_chunk_size or size
    elif dim == 'lead_time':
      stat_chunks[dim] = times.lead_time_chunk_size or size
    else:
      stat_chunks[dim] = size  # unchunked

  if zarr_chunks is None:
    zarr_chunks = stat_chunks.copy()
  else:
    zarr_chunks = {**stat_chunks, **zarr_chunks}

  def process_and_store_chunk(
      chunk_index: int,
      init_chunk: np.ndarray,
      lead_chunk: np.ndarray,
  ) -> xr.Dataset:
    """Process a chunk and return the statistics dataset."""
    targets = targets_loader.load_chunk(init_chunk, lead_chunk)
    predictions = predictions_loader.load_chunk(init_chunk, lead_chunk, targets)
    stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, predictions, targets
    )
    chunk_ds = xr.Dataset()
    for stat_name, var_dict in stats.items():
      for var_name, da in var_dict.items():
        chunk_ds[f'{stat_name}.{var_name}'] = da
    return chunk_ds

  # Process chunks and collect results
  chunks = list(times)
  n_chunks = len(chunks)
  logging.info(f'Processing {n_chunks} chunks for unaggregated output...')

  all_datasets = []

  if n_workers > 1:
    @dask.delayed
    def delayed_process(i, init_chunk, lead_chunk):
      return process_and_store_chunk(i, init_chunk, lead_chunk)

    delayed_results = [
        delayed_process(i, init_chunk, lead_chunk)
        for i, (init_chunk, lead_chunk) in enumerate(chunks)
    ]

    with dask.config.set(num_workers=n_workers):
      if show_progress:
        with ProgressBar():
          all_datasets = list(dask.compute(*delayed_results))
      else:
        all_datasets = list(dask.compute(*delayed_results))
  else:
    for i, (init_chunk, lead_chunk) in enumerate(chunks):
      if show_progress:
        logging.info(f'Processing chunk {i + 1}/{n_chunks}...')
      ds = process_and_store_chunk(i, init_chunk, lead_chunk)
      all_datasets.append(ds)

  # Combine all datasets
  logging.info('Combining datasets...')
  combined = xr.combine_by_coords(all_datasets)

  # Rechunk to target chunks
  logging.info(f'Writing to {out_path}...')
  combined = combined.chunk(zarr_chunks)
  combined.to_zarr(out_path, mode='w')
  logging.info('Done!')


# Alias for backwards compatibility with beam_pipeline interface
define_pipeline = run_pipeline
define_unaggregated_pipeline = run_unaggregated_pipeline
