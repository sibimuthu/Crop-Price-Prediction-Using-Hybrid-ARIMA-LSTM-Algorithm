import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 6, 1)
    days = (end_date - start_date).days
    
    date_range = [start_date + timedelta(days=x) for x in range(days)]
    
    data = []
    
    # Parameters for synthetic generation
    crops = {
        'Tomato': {'base': 40, 'amp': 20, 'noise': 5, 'season': 180},
        'Onion': {'base': 30, 'amp': 15, 'noise': 4, 'season': 365},
        'Potato': {'base': 20, 'amp': 5, 'noise': 2, 'season': 365},
    }
    
    for crop, params in crops.items():
        base = params['base']
        amp = params['amp']
        noise = params['noise']
        season = params['season']
        
        for i, date in enumerate(date_range):
            # Sinusoidal Seasonality
            seasonal_effect = amp * np.sin(2 * np.pi * i / season)
            random_noise = np.random.normal(0, noise)
            
            price = base + seasonal_effect + random_noise
            price = max(5, price) # Ensure positive price
            
            data.append({
                'Date': date,
                'Crop': crop,
                'Price': round(price, 2),
                'Yield': 0, # Placeholder
                'Season': 'Synthetic'
            })
            
    df = pd.DataFrame(data)
    
    output_path = os.path.join('data', 'processed', 'merged_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic data: {len(df)} rows to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
