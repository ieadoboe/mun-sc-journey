import cdsapi

years = ['2015', '2016', '2017', '2018',
         '2019', '2020', '2021', '2022', '2023']

for year in years:
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_pressure",
            "surface_solar_radiation_downwards",
            "surface_sensible_heat_flux",
            "surface_latent_heat_flux",
        ],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "06:00", "12:00",
            "18:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [60, -140, 40, -50]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(
        target=f'era5_surface_{year}.grib')

    print(f"  {year} surface download complete!")

print(f"All {len(years)} years complete!")
print("Surface download complete!")
