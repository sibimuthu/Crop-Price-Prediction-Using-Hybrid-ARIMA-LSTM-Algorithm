import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Import functions from previous modules to get predictions
# Note: In a production system, we would load saved predictions/models.
# For this script, we will import and run specific functions or assume consistent data splits.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arima_model import train_sarima, evaluate_model
from models.lstm_model import WheatLSTM, create_sequences, device
import torch
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    subset = df[df['Crop'] == 'Wheat'].copy()
    daily_price = subset.groupby('Date')['Price'].mean().reset_index()
    daily_price = daily_price.set_index('Date').asfreq('D').ffill()
    return daily_price['Price']

def get_sarima_preds(data):
    # Same split as before
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:]
    
    # Train SARIMA (using same params as arima_model.py)
    # Seasonal order (1, 1, 1, 7)
    print("Generating SARIMA predictions...")
    model = train_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    
    # Forecast
    start = len(train)
    end = len(train) + len(test) - 1
    preds = model.predict(start=start, end=end, typ='levels')
    return preds, test

def get_lstm_preds(data):
    # Same preprocessing as lstm_model.py
    raw = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)
    
    SEQ_LENGTH = 30
    X, y = create_sequences(scaled, SEQ_LENGTH)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    
    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures', 'lstm_model.pth')
    model = WheatLSTM(1, 50, 1, 1).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print("Warning: Could not load saved LSTM model. Results might be garbage.")
    
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).cpu().numpy()
        
    preds = scaler.inverse_transform(preds_scaled).flatten()
    
    # Align LSTM preds with original index
    # LSTM trims raw data by SEQ_LENGTH
    # The test set starts at train_size index OF THE SEQUENCES
    # Original Data Index: [0 ... SEQ ... Train_End ... Test_End]
    # Sequence Data Index: [0 ... Train_End-SEQ ... ]
    
    # Effectively, LSTM test predictions correspond to data[SEQ_LENGTH + train_size : ] ?? 
    # Let's align carefully.
    # X[0] uses data[0:30] to predict data[30]
    # X[train_size] is the first test sequence.
    # It predicts y[train_size], which is data[train_size + SEQ_LENGTH]
    
    start_idx = train_size + SEQ_LENGTH
    params_idx = data.index[start_idx : start_idx + len(preds)]
    
    return pd.Series(preds, index=params_idx)

def run_hybrid_logic():
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'merged_data.csv')
    data = load_data(data_path)
    
    # 1. Get Predictions
    sarima_preds, sarima_actuals = get_sarima_preds(data)
    lstm_preds = get_lstm_preds(data)
    
    # 2. Align Indices
    # SARIMA test set: data[train_size:]
    # LSTM test set: data[train_size + SEQ:]
    # Intersection is the LSTM test set range.
    
    common_idx = lstm_preds.index.intersection(sarima_preds.index)
    
    s_preds = sarima_preds.loc[common_idx]
    l_preds = lstm_preds.loc[common_idx]
    actuals = data.loc[common_idx]
    
    print(f"\nAligned Test Set Size: {len(common_idx)}")
    
    # 3. Hybrid Strategy: Simple Average
    hybrid_preds = (s_preds + l_preds) / 2
    
    # 4. Evaluation
    metrics = {}
    for name, pred in [('SARIMA', s_preds), ('LSTM', l_preds), ('Hybrid', hybrid_preds)]:
        mse = mean_squared_error(actuals, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, pred)
        r2 = r2_score(actuals, pred)
        mape = mean_absolute_percentage_error(actuals, pred)
        accuracy = 100 * (1 - mape)
        
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'R2': r2, 'Accuracy': accuracy}
        print(f"{name} -> RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Accuracy: {accuracy:.2f}%")
        
    # 5. Plot Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(actuals.index, actuals, label='Actual', color='black', alpha=0.5)
    plt.plot(s_preds.index, s_preds, label='SARIMA', linestyle='--')
    plt.plot(l_preds.index, l_preds, label='LSTM', linestyle='--')
    plt.plot(hybrid_preds.index, hybrid_preds, label='Hybrid (Ensemble)', linewidth=2, color='red')
    plt.title('Hybrid Model vs Individual Models (Wheat Price)')
    plt.legend()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
    plt.savefig(os.path.join(output_dir, 'hybrid_comparison.png'))
    print(f"Saved comparison plot to {output_dir}")

if __name__ == "__main__":
    run_hybrid_logic()
