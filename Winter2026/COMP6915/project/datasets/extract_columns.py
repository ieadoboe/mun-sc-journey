import xarray as xr

pressure_data = xr.open_dataset('era5_pressure_levels.nc')
surface_data = xr.open_dataset('era5_surface.nc')

locations = {
    'St_Johns': (47.6, -52.7),
    'Toronto': (43.7, -79.4),
    'Vancouver': (49.3, -123.1),
}

for loc_name, (lat, lon) in locations.items():
    p_col = pressure_data.sel(latitude=lat, longitude=lon, method='nearest')
    s_col = surface_data.sel(latitude=lat, longitude=lon, method='nearest')
    column = xr.merge([p_col, s_col])
    column.to_netcdf(f'era5_column_{loc_name}.nc')
    print(f"✓ Extracted {loc_name}")
