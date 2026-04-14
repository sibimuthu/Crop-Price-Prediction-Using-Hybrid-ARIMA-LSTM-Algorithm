import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def plot_price_trends(df, crop, output_dir):
    """Plots price trends over time."""
    plt.figure(figsize=(14, 7))
    subset = df[df['Crop'] == crop]
    # Aggregate to daily average for clearer trend line
    daily_avg = subset.groupby('Date')['Price'].mean().reset_index()
    sns.lineplot(data=subset, x='Date', y='Price', label='Raw Data (Districts)', alpha=0.3)
    plt.plot(daily_avg['Date'], daily_avg['Price'], label='Daily Average', color='blue', linewidth=2)
    plt.title(f'Price Trend for {crop}')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs/kg)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{crop}_price_trend.png'))
    plt.close()

def plot_yield_trends(df, crop, output_dir):
    """Plots yield trends over years."""
    plt.figure(figsize=(10, 6))
    subset = df[df['Crop'] == crop].drop_duplicates(subset=['Year'])
    subset = subset.sort_values(by='Year')
    plt.plot(subset['Year'], subset['Yield'], marker='o', color='green')
    plt.title(f'Yield Trend for {crop}')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{crop}_yield_trend.png'))
    plt.close()

def plot_seasonal_decomposition(df, crop, output_dir):
    """Decomposes time series into Trend, Seasonal, and Residual."""
    subset = df[df['Crop'] == crop].copy()
    
    # Aggregate duplicates (mean price per day)
    subset = subset.groupby('Date')['Price'].mean().reset_index()
    
    subset.set_index('Date', inplace=True)
    subset = subset.sort_index()
    
    # Need regular intervals for decomposition. Resample to Daily.
    subset = subset.resample('D').ffill()
    
    try:
        # Period = 365 for daily data yearly seasonality
        result = seasonal_decompose(subset['Price'], model='additive', period=365)
        
        plt.figure(figsize=(14, 10))
        result.plot()
        plt.suptitle(f'Seasonal Decomposition for {crop}', y=1.02)
        plt.savefig(os.path.join(output_dir, f'{crop}_seasonal_decompose.png'))
        plt.close()
    except Exception as e:
        print(f"Could not decompose for {crop}: {e}")

def plot_correlation(df, crop, output_dir):
    """Plots correlation heatmap between Price and other features."""
    subset = df[df['Crop'] == crop]
    cols = ['Price', 'Yield', 'T2M', 'PRECTOTCORR', 'RH2M']
    # Filter only existing columns
    cols = [c for c in cols if c in subset.columns]
    
    corr = subset[cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for {crop}')
    plt.savefig(os.path.join(output_dir, f'{crop}_correlation.png'))
    plt.close()

def generate_eda_report(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    for crop in df['Crop'].unique():
        print(f"Generating plots for {crop}...")
        plot_price_trends(df, crop, output_dir)
        plot_yield_trends(df, crop, output_dir)
        plot_seasonal_decomposition(df, crop, output_dir)
        plot_correlation(df, crop, output_dir)
    print("EDA Generation Complete.")

if __name__ == "__main__":
    import sys
    # Default paths
    input_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'merged_data.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
    generate_eda_report(input_path, output_path)
