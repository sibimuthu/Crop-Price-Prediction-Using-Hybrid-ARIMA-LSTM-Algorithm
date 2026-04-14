import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import loader
from src.data import preprocessing

def main():
    print("Step 1: Loading Data...")
    raw_data = loader.get_complete_data_dict()
    
    print("\nStep 2: Processing and Merging...")
    processed_df = preprocessing.process_and_merge(raw_data)
    
    print("\nStep 3: Post-Processing Checks...")
    print(f"Final Info:\n{processed_df.info()}")
    print(f"\nMissing Values:\n{processed_df.isnull().sum()}")
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'merged_data.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    
    # Crop Summary
    for crop in processed_df['Crop'].unique():
        print(f"\nStats for {crop}:")
        print(processed_df[processed_df['Crop'] == crop][['Price', 'Yield']].describe())

if __name__ == "__main__":
    main()
