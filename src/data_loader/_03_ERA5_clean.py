import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / 'data' / 'processed' / 'dynamic_feature.csv'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed' / 'climate_feature.csv'

CLIMATE_1624_PATH = BASE_DIR / 'data' / 'raw' / 'EU(16-24)' / 'data.grib'
CLIMATE_0015_PATH = BASE_DIR / 'data' / 'raw' / 'EU(00-15)' / 'data.grib'

def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_DIR)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df.rename(columns={'ISO_TIME': 'DATETIME'}, inplace=True)
    return df

def split_data_by_year(df: pd.DataFrame, year_ranges: list = None) -> dict:
    if year_ranges is None:
        year_ranges = [(2000, 2015), (2016, 2024)]
    
    result = {}
    for i, (start, end) in enumerate(year_ranges):
        key = f"df_{start}-{end}"
        result[key] = df[(df['YEAR'] >= start) & (df['YEAR'] <= end)].copy()
    
    return result


def load_grib_dataset(grib_path: str, engine: str = "cfgrib") -> xr.Dataset:
    return xr.open_dataset(grib_path, engine=engine)


def extract_era5_features(
    df: pd.DataFrame,
    ds: xr.Dataset
) -> pd.DataFrame:
    # Create query DataArrays
    query_time = xr.DataArray(df['DATETIME'].values, dims="points")
    query_lat = xr.DataArray(df['LAT'].values, dims="points")
    query_lon = xr.DataArray(df['LON'].values, dims="points")
    
    # Extract nearest values
    ds_extract = ds.sel(
        time=query_time,
        latitude=query_lat,
        longitude=query_lon,
        method="nearest"
    )
    
    # Convert to DataFrame
    df_grib_features = ds_extract.to_dataframe().reset_index()
    
    # Extract datetime components
    df_grib_features['YEAR'] = df_grib_features['time'].dt.year
    df_grib_features['MONTH'] = df_grib_features['time'].dt.month
    df_grib_features['DAY'] = df_grib_features['time'].dt.day
    df_grib_features['HOUR'] = df_grib_features['time'].dt.hour
    
    # Select relevant columns
    var_names = list(ds.data_vars)
    cols = ['YEAR', 'MONTH', 'DAY', 'HOUR'] + var_names
    df_final = df_grib_features[cols]
    
    print(f"Đã trích xuất thành công {len(df_final)} dòng dữ liệu.")
    return df_final


def combine_with_coordinates(
    df_coords: pd.DataFrame,
    df_features: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat([df_coords, df_features], axis=1)


def process_climate_data(data: pd.DataFrame) -> pd.DataFrame:
    # Convert mean sea level pressure to hPa
    data['msl'] = data['msl'] / 100
    
    # Interpolate and convert sea surface temperature from Kelvin to Celsius
    data['sst'] = data['sst'].interpolate(method='linear')
    data['sst'] = data['sst'] - 273.15
    
    return data


def ERA5_extracting():
    # Load feature data
    df = load_feature_data()
    
    # Split data by year
    data_splits = split_data_by_year(df)
    
    # Load GRIB datasets
    ds_1624 = load_grib_dataset(CLIMATE_1624_PATH)
    ds_0015 = load_grib_dataset(CLIMATE_0015_PATH)
    
    # Extract features for each period
    df_1624 = data_splits['df_2016-2024']
    df_0015 = data_splits['df_2000-2015']
    
    lat_lon_1624 = df_1624[['LAT', 'LON']].reset_index().drop(columns='index')
    lat_lon_0015 = df_0015[['LAT', 'LON']].reset_index().drop(columns='index')
    
    # Extract ERA5 features
    df_final_1624 = extract_era5_features(df_1624, ds_1624)
    df_final_0015 = extract_era5_features(df_0015, ds_0015)
    
    # Combine with coordinates
    data_1624 = combine_with_coordinates(lat_lon_1624, df_final_1624)
    data_0015 = combine_with_coordinates(lat_lon_0015, df_final_0015)
    
    # Concatenate all data
    data = pd.concat([data_0015, data_1624], axis=0)
    
    # Process climate data
    data = process_climate_data(data)
    
    # Save to CSV
    data.to_csv(OUTPUT_DIR, index=False)
    print(f"Dữ liệu đã được lưu vào {OUTPUT_DIR}")


if __name__ == "__main__":
    ERA5_extracting()