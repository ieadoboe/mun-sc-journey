import xarray as xr
import glob

# ============================================================================
# STEP 1: Combine all yearly GRIB files
# ============================================================================

print("Combining pressure-level data...")
pressure_files = sorted(glob.glob('era5_pressure_*.grib'))
pressure_data = xr.open_mfdataset(
    pressure_files, engine='cfgrib', combine='by_coords')
pressure_data.to_netcdf('era5_pressure_2015_2023.nc')
print(f"  Combined {len(pressure_files)} pressure files")

print("\nCombining surface data...")
surface_files = sorted(glob.glob('era5_surface_*.grib'))
surface_data = xr.open_mfdataset(
    surface_files, engine='cfgrib', combine='by_coords')
surface_data.to_netcdf('era5_surface_2015_2023.nc')
print(f"  Combined {len(surface_files)} surface files")

# ============================================================================
# STEP 2: Explore data structure
# ============================================================================

print("\n" + "="*70)
print("PRESSURE DATA STRUCTURE")
print("="*70)
print(pressure_data)
print(f"\nVariables: {list(pressure_data.data_vars)}")
print(f"Coordinates: {list(pressure_data.coords)}")
print(f"Shape: {pressure_data['t'].shape}")  # Temperature shape

print("\n" + "="*70)
print("SURFACE DATA STRUCTURE")
print("="*70)
print(surface_data)
print(f"\nVariables: {list(surface_data.data_vars)}")
print(f"Coordinates: {list(surface_data.coords)}")
print(f"Shape: {surface_data['t2m'].shape}")  # 2m temperature shape
