import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple

def convert_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Price from Rs/Quintal to Rs/Kg.
    Formula: Price_Kg = Price_Quintal / 100
    """
    if 'Price' in df.columns:
        df['Price'] = df['Price'] / 100.0
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values using Forward Fill and then Backward Fill.
    """
    # Sort by Date if present
    if 'Date' in df.columns:
        df = df.sort_values(by='Date')
        
    df = df.ffill().bfill()
    return df

def remove_outliers(df: pd.DataFrame, column: str = 'Price', method: str = 'IQR') -> pd.DataFrame:
    """
    Removes outliers using IQR method.
    """
    if column not in df.columns:
        return df
        
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed {len(df) - len(df_clean)} outliers from {column}")
    return df_clean

def normalize_features(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Normalizes specified columns using MinMaxScaler.
    Returns the dataframe and a dictionary of scalers (one per column or global).
    """
    scalers = {}
    for col in columns:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
    return df, scalers

def merge_datasets(price_df: pd.DataFrame, yield_df: pd.DataFrame, 
                   weather_df: pd.DataFrame, seasonal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges all datasets into a single daily time-series dataframe per crop.
    Strategy:
    1. Base: Price DF (Daily).
    2. Seasonal: Merge on Date.
    3. Weather: Merge on Date (Daily weather has been interpolated/forward filled in loader or here).
       Note: Loader produces weather with 'Date'. We can merge on Date directly if it matches, 
       otherwise we might need to ffill weather from the nearest previous month start.
    4. Yield: Yield is annual/seasonal. Merge on Year/Season if possible, or just Year.
    """
    
    # Ensure Dates
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    seasonal_df['Date'] = pd.to_datetime(seasonal_df['Date'])
    
    # 1. Merge Seasonal
    # Seasonal DF likely has all dates. Left join price to seasonal to keep price dates?
    # Or strict join?
    # Better: Join Price with Seasonal on Date.
    merged = pd.merge(price_df, seasonal_df[['Date', 'Season']], on='Date', how='left')
    
    # 2. Merge Weather
    # Weather is monthly (Date is 1st of month). We need to ffill weather for the rest of the month.
    # We can perform an 'asof' merge or reindex weather to daily.
    weather_daily = weather_df.set_index('Date').resample('D').ffill().reset_index()
    merged = pd.merge(merged, weather_daily, on='Date', how='left')
    
    # 3. Merge Yield
    # Yield has 'Crop_Year' and 'Season'.
    # Price data has 'Date', so we can derive Year.
    merged['Year'] = merged['Date'].dt.year
    
    # For yield, we might need to aggregate by Year and Crop to get a broad yield metric for the join.
    # Since Yield is state-wise and Price is district-wise (or state-wise), this is complex.
    # Simplification: Average Yield per Year per Crop across all states (or filter if Price has State).
    # Let's assume we want a general Yield indicator for the Crop for that Year.
    
    # Aggregate yield by Year and Crop
    if not yield_df.empty:
        yield_agg = yield_df.groupby(['Crop', 'Crop_Year'])['Yield'].mean().reset_index()
        yield_agg.rename(columns={'Crop_Year': 'Year'}, inplace=True)
        
        merged = pd.merge(merged, yield_agg, on=['Crop', 'Year'], how='left')
        
        # Impute missing Yield (for years > 2020) using historical average per crop
        for crop in merged['Crop'].unique():
            # Calculate mean yield for this crop from available data
            avg_yield = yield_agg[yield_agg['Crop'] == crop]['Yield'].mean()
            if not pd.isna(avg_yield):
                # Fill NaNs in Yield column for this crop
                # Using boolean indexing for assignment
                mask = (merged['Crop'] == crop) & (merged['Yield'].isna())
                merged.loc[mask, 'Yield'] = avg_yield
    else:
        merged['Yield'] = 0 # Placeholder if no yield data
        
    return merged

def process_and_merge(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Main pipeline to process and merge data.
    """
    price_df = data_dict['price']
    yield_df = data_dict['yield']
    weather_df = data_dict['weather']
    seasonal_df = data_dict['seasonal']
    
    # 1. Price Processing
    price_df = convert_price(price_df)
    
    # Outlier removal (e.g. Price > 0)
    price_df = price_df[price_df['Price'] > 0]
    # We can apply IQR per crop
    processed_prices = []
    for crop in price_df['Crop'].unique():
        crop_df = price_df[price_df['Crop'] == crop].copy()
        crop_df = remove_outliers(crop_df, 'Price')
        processed_prices.append(crop_df)
        
    if processed_prices:
        price_df = pd.concat(processed_prices)
        
    # 2. Merge
    full_df = merge_datasets(price_df, yield_df, weather_df, seasonal_df)
    
    # 3. Clean Merged Data (Impute missing Yield/Weather)
    full_df = clean_data(full_df)
    
    return full_df
