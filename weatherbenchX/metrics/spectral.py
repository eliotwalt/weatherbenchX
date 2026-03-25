
import dataclasses
import numpy as np
import xarray as xr

@dataclasses.dataclass
class ZonalEnergySpectrum:
  """Energy spectrum along the zonal direction.

  Given dataset with longitude dimension, this class computes spectral energy as
  a function of wavenumber (as a dim). wavelength and frequency are also present
  as coords with units "1 / m" and "m" respectively. Only non-negative
  frequencies are included.

  Let f[l], l = 0,..., L - 1, be dataset values along a zonal circle of constant
  latitude, with circumference C (m).  The DFT is
    F[k] = (1 / L) Σₗ f[l] exp(-i2πkl/L)
  The energy spectrum is then set to
    S[0] = C |F[0]|²,
    S[k] = 2 C |F[k]|², k > 0, to account for positive and negative frequencies.

  With C₀ the equatorial circumference, the ith zonal circle has circumference
    C(i) = C₀ Cos(π latitude[i] / 180).
  Since data points occur at longitudes longitude[l], l = 0, ..., L - 1, the DFT
  will measure spectra at zonal sampling frequencies
    f(k, i) = longitude[k] / (C(i) 360), k = 0, ..., L // 2,
  and corresponding wavelengths
    λ(k, i) = 1 / f(k, i).

  This choice of normalization ensures Parseval's relation for energy holds:
  Supposing f[l] are sampled values of f(ℓ), where 0 < ℓ < C (meters) is a
  coordinate on the circle. Then (C / L) is the spacing of longitudinal samples,
  whence
    ∫|f(ℓ)|² dℓ ≈ (C / L) Σₗ |f[l]|² = Σₖ S[k].

  If f has units β, then S has units of m β². For example, if f is
  `u_component_of_wind`, with units (m / s), then S has units (m³ / s²). In
  air with mass density ρ (kg / m³), this gives energy density at wavenumber k
    ρ S[k] ~ (kg / m³) (m³ / s²) = kg / s²,
  which is energy density (per unit area).
  """

  variable_name: str

  @property
  def base_variables(self) -> list[str]:
    return [self.variable_name]

  @property
  def core_dims(self):
    return (['longitude'],), ['zonal_wavenumber']

  def _circumference(self, dataset: xr.Dataset) -> xr.DataArray:
    """Earth's circumference as a function of latitude."""
    EARTH_RADIUS_M = 1000 * (6357 + 6378) / 2
    circum_at_equator = 2 * np.pi * EARTH_RADIUS_M
    return np.cos(dataset.latitude * np.pi / 180) * circum_at_equator

  def lon_spacing_m(self, dataset: xr.Dataset) -> xr.DataArray:
    """Spacing (meters) between longitudinal values in `dataset`."""
    diffs = dataset.longitude.diff('longitude')
    if np.max(np.abs(diffs - diffs[0])) > 1e-3:
      raise ValueError(
          f'Expected uniform longitude spacing. {dataset.longitude.values=}'
      )
    return self._circumference(dataset) * diffs[0].data / 360

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    """Computes zonal power at wavenumber and frequency."""
    spacing = self.lon_spacing_m(dataset)

    def simple_power(f_x):
      f_k = np.fft.rfft(f_x, axis=-1, norm='forward')
      # freq > 0 should be counted twice in power since it accounts for both
      # positive and negative complex values.
      one_and_many_twos = np.concatenate(([1], [2] * (f_k.shape[-1] - 1)))
      return np.real(f_k * np.conj(f_k)) * one_and_many_twos

    spectrum = xr.apply_ufunc(
        simple_power,
        dataset,
        input_core_dims=[['longitude']],
        output_core_dims=[['longitude']],
        exclude_dims={'longitude'},
    ).rename_dims({'longitude': 'zonal_wavenumber'})[self.variable_name]
    spectrum = spectrum.assign_coords(
        zonal_wavenumber=('zonal_wavenumber', spectrum.zonal_wavenumber.data)
    )
    base_frequency = xr.DataArray(
        np.fft.rfftfreq(len(dataset.longitude)),
        dims='zonal_wavenumber',
        coords={'zonal_wavenumber': spectrum.zonal_wavenumber},
    )
    spectrum = spectrum.assign_coords(frequency=base_frequency / spacing)
    spectrum['frequency'] = spectrum.frequency.assign_attrs(units='1 / m')

    spectrum = spectrum.assign_coords(wavelength=1 / spectrum.frequency)
    spectrum['wavelength'] = spectrum.wavelength.assign_attrs(units='m')

    # This last step ensures the sum of spectral components is equal to the
    # (discrete) integral of data around a line of latitude.
    return spectrum * self._circumference(spectrum)