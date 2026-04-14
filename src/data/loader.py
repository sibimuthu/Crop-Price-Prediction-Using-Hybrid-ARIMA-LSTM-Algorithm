import pandas as pd
import os
import glob
from typing import Dict, List, Optional

# Define constants for file paths based on user's workspace
PRICE_FILES = [
    r'c:/Users/Sibiraj/Downloads/Project Dataset/Crop Price Dataset/Crop Price Dataset-2.csv',
    r'c:/Users/Sibiraj/Downloads/Project Dataset/Crop Price Dataset/Crop Price Dataset-3.csv'
]
YIELD_FILE = r'c:/Users/Sibiraj/Downloads/Project Dataset/Crop Yield Dataset/Crop Yield Dataset-2.csv'
WEATHER_FILE = r'c:/Users/Sibiraj/Downloads/Project Dataset/Weather Dataset/Weather Dataset.csv'
SEASONAL_FILE = r'c:/Users/Sibiraj/Downloads/Project Dataset/Seasonal Dataset/seasonal dataset.csv'

TARGET_CROPS = ['Tomato', 'Onion', 'Potato']

def load_price_data(file_paths: List[str] = PRICE_FILES) -> pd.DataFrame:
    """
    Loads and consolidates price data from multiple CSVs.
    Normalizes column names and interprets dates.
    """
    dfs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"Warning: Price file not found: {fp}")
            continue
            
        try:
            df = pd.read_csv(fp)
            
            # Normalize Columns
            cols_map = {
                'Price Date': 'Date', 'Arrival_Date': 'Date',
                'Modal Price (Rs./Quintal)': 'Price', 'Modal Price': 'Price',
                'Commodity': 'Crop',
                'District Name': 'District',
                'State': 'State'
            }
            df = df.rename(columns=cols_map)
            
            # Filter for target crops
            if 'Crop' in df.columns:
                # Case insensitive filter
                mask = df['Crop'].str.contains('|'.join(TARGET_CROPS), case=False, na=False)
                df = df[mask]
                
                # Standardize Crop names
                for crop in TARGET_CROPS:
                    df.loc[df['Crop'].str.contains(crop, case=False, na=False), 'Crop'] = crop
            
            # Select only relevant columns
            keep_cols = ['Date', 'Crop', 'State', 'District', 'Price']
            df = df[[c for c in keep_cols if c in df.columns]]
            
            # Convert Date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {os.path.basename(fp)}")
            
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.dropna(subset=['Date', 'Price'])
    final_df = final_df.sort_values(by='Date')
    return final_df

def load_yield_data(file_path: str = YIELD_FILE) -> pd.DataFrame:
    """
    Loads yield data and filters for target crops.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Yield file not found: {file_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        
        # Filter Crops
        mask = df['Crop'].str.contains('|'.join(TARGET_CROPS), case=False, na=False)
        df = df[mask]
        
        # Standardize Crop names
        for crop in TARGET_CROPS:
            df.loc[df['Crop'].str.contains(crop, case=False, na=False), 'Crop'] = crop
            
        print(f"Loaded {len(df)} yield records.")
        return df
    except Exception as e:
        print(f"Error loading yield data: {e}")
        return pd.DataFrame()

def load_weather_data(file_path: str = WEATHER_FILE) -> pd.DataFrame:
    """
    Loads weather data and reshapes from monthly wide format to daily/monthly long format.
    Assumes structure: PARAMETER, YEAR, JAN, FEB...
    """
    if not os.path.exists(file_path):
        print(f"Warning: Weather file not found: {file_path}")
        return pd.DataFrame()
        
    try:
        # Skip header rows if present (based on view_file output, header is ~11 lines starting with -BEGIN-)
        # But pandas read_csv might need 'header' arg. 
        # Looking at file content: Line 12 is header: PARAMETER,YEAR,JAN...
        df = pd.read_csv(file_path, skiprows=11) 
        
        # Melting months
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        melted = df.melt(id_vars=['PARAMETER', 'YEAR'], 
                         value_vars=list(month_map.keys()), 
                         var_name='Month_Name', value_name='Value')
        
        melted['Month'] = melted['Month_Name'].map(month_map)
        
        # Create Date (set to 1st of month)
        melted['Date'] = pd.to_datetime(melted.assign(Day=1)[['YEAR', 'Month', 'Day']])
        
        # Pivot to get Parameters as columns
        weather_df = melted.pivot_table(index='Date', columns='PARAMETER', values='Value').reset_index()
        
        print(f"Loaded weather data: {len(weather_df)} monthly records.")
        return weather_df
        
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return pd.DataFrame()

def load_seasonal_data(file_path: str = SEASONAL_FILE) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading seasonal data: {e}")
        return pd.DataFrame()

def get_complete_data_dict():
    """
    Master function to get all data.
    """
    print("Loading Price Data...")
    price_df = load_price_data()
    
    print("Loading Yield Data...")
    yield_df = load_yield_data()
    
    print("Loading Weather Data...")
    weather_df = load_weather_data()
    
    print("Loading Seasonal Data...")
    seasonal_df = load_seasonal_data()
    
    return {
        'price': price_df,
        'yield': yield_df,
        'weather': weather_df,
        'seasonal': seasonal_df
    }

if __name__ == "__main__":
    # Test loading
    data = get_complete_data_dict()
    for k, v in data.items():
        print(f"\ndataset: {k}")
        print(v.head())
        print(v.info())
