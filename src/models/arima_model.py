import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

def check_stationarity(timeseries):
    """
    Performs Augmented Dickey-Fuller test to check for stationarity.
    Returns True if stationary (p-value < 0.05), else False.
    """
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return dfoutput['p-value'] < 0.05

def train_arima(train_data, order=(5,1,0)):
    """
    Trains an ARIMA model.
    """
    print(f"Training ARIMA with order {order}...")
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def train_sarima(train_data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Trains a SARIMA model.
    """
    print(f"Training SARIMA with order {order} and seasonal_order {seasonal_order}...")
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    return model_fit

def evaluate_model(model_fit, test_data):
    """
    Evaluates the model on test data using forecasts.
    """
    history = [x for x in model_fit.data.endog]
    predictions = []
    for t in range(len(test_data)):
        # One-step ahead forecast
        # For simplicity in this script using built-in forecast but for rolling origin:
        # Re-training/updating is expensive. We'll use the predict method on the fitted model 
        # extended by observations? Or just forecast from end of train.
        # Let's simple forecast strictly from end of train for now.
        
        # Better: use get_forecast on the result object if it supports strict out-of-sample
        pass
    
    # Simple static forecast for evaluation
    start = len(history)
    end = len(history) + len(test_data) - 1
    predictions = model_fit.predict(start=start, end=end, typ='levels')
    
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mse)
    
    print(f'Model Evaluation:\nRMSE: {rmse}\nMAE: {mae}')
    return predictions, rmse, mae

def plot_forecast(train, test, forecast, label, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'{label} Forecast vs Actuals')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_forecast.png'))
    plt.close()

def run_pipeline(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Process for specific crop: Wheat
    crop = 'Wheat'
    subset = df[df['Crop'] == crop].copy()
    
    # Aggregate to daily mean price
    daily_df = subset.groupby('Date')['Price'].mean()
    daily_df = daily_df.asfreq('D').ffill()
    
    print(f"Data shape for {crop}: {daily_df.shape}")
    
    # 1. Stationarity Check
    is_stationary = check_stationarity(daily_df)
    d = 0 if is_stationary else 1
    print(f"Stationarity: {is_stationary}, Suggested differencing (d): {d}")
    
    # Split Train/Test
    train_size = int(len(daily_df) * 0.8)
    train, test = daily_df[0:train_size], daily_df[train_size:]
    
    # 2. ARIMA
    # Simple heuristic for p, q. Auto_arima would be better but keeping it simple/manual for now.
    # d is determined. Let's try (5, d, 2)
    arima_order = (5, d, 0) 
    arima_model = train_arima(train, order=arima_order)
    arima_pred, arima_rmse, _ = evaluate_model(arima_model, test)
    plot_forecast(train, test, arima_pred, 'ARIMA', output_dir)
    
    # 3. SARIMA
    # Seasonal period for daily data could be 7 (weekly) or 30 (monthly) or 365 (yearly).
    # 365 is heavy. Let's try 7 for weekly seasonality first.
    seasonal_order = (1, 1, 1, 7)
    sarima_model = train_sarima(train, order=(1, d, 1), seasonal_order=seasonal_order)
    sarima_pred, sarima_rmse, _ = evaluate_model(sarima_model, test)
    plot_forecast(train, test, sarima_pred, 'SARIMA', output_dir)
    
    print("Modeling Complete.")

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'merged_data.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
    run_pipeline(input_path, output_path)
