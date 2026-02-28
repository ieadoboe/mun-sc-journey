import cdsapi
years = ['2015', '2016', '2017', '2018',
         '2019', '2020', '2021', '2022', '2023']

for year in years:
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "specific_humidity",
            "temperature"
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
        "pressure_level": [
            "300", "500", "700",
            "850", "975", "1000"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        # North=60°N, West=140°W, South=40°N, East=50°W = Canada
        "area": [60, -140, 40, -50]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(
        target=f'era5_pressure_{year}.grib')
    print(f"  {year} pressure download complete!")

print(f"All {len(years)} years complete!")
print("Pressure-level download complete!")
