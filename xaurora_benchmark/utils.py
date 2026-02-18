import numpy as np
from scipy.ndimage import convolve
from absl import logging
import xarray as xr

def convolve_fill_nan(da):
    """Fill NaN values in a DataArray using a weighted convolution of neighboring values.
    
    Weights as in GenCast paper (Section A.2.1)"""
    kernel = np.array([[0.5, 1.0, 0.5],
                        [1.0, 0.0, 1.0],
                        [0.5, 1.0, 0.5]])
    
    # make sure lat/lon are the last two dimensions of da
    da = da.transpose(..., 'latitude', 'longitude')
    
    data = da.values
    
    # add dummy dimensions for convolution if needed
    ndims = data.ndim
    kernel = kernel.reshape(*(
        [1 for _ in range(ndims - 2)] + [3, 3]
    )) # (..., 3, 3)

    # Mask of valid values
    valid = np.isfinite(data)
    
    if np.any(~valid):        
        logging.info(f"Filling {np.sum(~valid)} NaNs in variable {da.name}")
        # Replace NaN with 0 for convolution
        filled_zero = np.where(valid, data, 0.0)
        # Convolve data
        weighted_sum = convolve(filled_zero, kernel, mode="constant", cval=0.0)
        # Convolve mask to get effective weight sum
        weight_sum = convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
        # Avoid division by zero
        result = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
        # Only replace NaNs
        final = np.where(valid, data, result)
    else:
        final = data

    return xr.DataArray(final, coords=da.coords, dims=da.dims, attrs=da.attrs)

def apply_convolve_fill_nan(ds):
    """Apply convolve_fill_nan to all variables in the dataset.
    
    Use the same weight matrix as in GenCast paper (Section A.2.1)
    """
    
    logging.info(f"Applying convolve_fill_nan to dataset with variables: {list(ds.data_vars)}")
    
    filled = {}
    for var in ds.data_vars:
        filled[var] = convolve_fill_nan(ds[var])
    return xr.Dataset(filled, coords=ds.coords, attrs=ds.attrs)