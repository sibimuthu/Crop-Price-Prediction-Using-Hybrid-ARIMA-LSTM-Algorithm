import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WheatLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WheatLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2) # Added dropout
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model(model, train_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    model.train()
    loss_hist = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if scheduler:
            scheduler.step(avg_loss)
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        loss_hist.append(avg_loss)
    return loss_hist

def evaluate(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            predictions.extend(out.cpu().numpy())
            actuals.extend(y_batch.numpy())
            
    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions))
    actuals = scaler.inverse_transform(np.array(actuals))
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    return actuals, predictions, rmse, mae

def run_lstm_pipeline(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    crops = df['Crop'].unique()
    if len(crops) == 0:
        print("No crops found in dataset.")
        return

    for crop in crops:
        print(f"\n--- Training LSTM for {crop} ---")
        # Filter for Crop and Daily Average
        subset = df[df['Crop'] == crop].copy()
        
        if len(subset) < 60:
            print(f"Skipping {crop}: Insufficient data ({len(subset)} records).")
            continue
            
        daily_price = subset.groupby('Date')['Price'].mean().reset_index()
        daily_price = daily_price.set_index('Date').asfreq('D').ffill()
        
        # Scale Data
        raw_data = daily_price['Price'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(raw_data)
        
        # Sequence Creation
        SEQ_LENGTH = 60 # Increased from 30 for more context
        if len(scaled_data) <= SEQ_LENGTH + 10:
             print(f"Skipping {crop}: Data too short for sequences.")
             continue
             
        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        
        # Train/Test Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # Data Loader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model Init
        input_size = 1 
        hidden_size = 128  # Increased from 50
        num_layers = 2     # Increased from 1
        output_size = 1
        
        model = WheatLSTM(input_size, hidden_size, num_layers, output_size).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        print("Starting Training...")
        train_loss_hist = train_model(model, train_loader, criterion, optimizer, num_epochs=100) # Increased epochs
        
        # Scheduler Step (using mean loss of last epoch)
        # However, scheduler is better placed inside train_model loop or called after each epoch with val loss.
        # Since we don't have separate val set in loop, we'll rely on training loss or integrate scheduler into train_model.
        
        # Let's simple pass scheduler to train_model? Or just stick to Adam for now with more capacity.
        # Actually, let's redefine train_model to take scheduler.
        
        print("Evaluating...")
        actuals, preds, rmse, mae = evaluate(model, test_loader, scaler)
        print(f"LSTM Results for {crop} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual Price')
        plt.plot(preds, label='LSTM Forecast', color='orange')
        plt.title(f'{crop} Price Prediction (LSTM)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'lstm_forecast_{crop}.png'))
        plt.close()
        
        # Save Model
        model_filename = f'lstm_model_{crop}.pth'
        torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
        print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'merged_data.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
    run_lstm_pipeline(input_path, output_path)
