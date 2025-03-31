import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_sequence(data, target_col='Close', seq_len=30):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data.iloc[i:i+seq_len].drop(columns=target_col).values)
            y.append(data.iloc[i+seq_len][target_col])
        return np.array(X, dtype='float32'), np.array(y, dtype='float32')

def get_predictions(ticker):


    df=pd.read_csv(f"df_for_gru_scaled_{ticker}.csv")
    df.dropna(inplace=True)
    df=df.astype('float32')
    
    X, y = create_sequence(df, target_col='Close', seq_len=30)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model_path = f"gru_model_{ticker}.keras"
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Loaded pretrained model for {ticker}")
    else:
        print(f"Model not found for {ticker}, training now...")
        train_and_save_model(ticker)
        model = load_model(model_path)

    # # Build model
    # model = Sequential()
    # model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dense(1)) 
    # model.compile(optimizer='adam', loss='mse')
    # # Train the model
    # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    #predict
    y_pred = model.predict(X_test)

    #inverse trtansform
    scaler = joblib.load(f"scaler_{ticker}.save")
    dummy_pred = np.zeros((len(y_pred), df.shape[1]))
    dummy_true = np.zeros((len(y_test), df.shape[1]))
    close_idx = df.columns.get_loc('Close')

    dummy_pred[:, close_idx] = y_pred[:, 0]
    dummy_true[:, close_idx] = y_test
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, close_idx]
    y_test_inv = scaler.inverse_transform(dummy_true)[:, close_idx]


    rmse = mean_squared_error(y_test_inv, y_pred_inv) ** 0.5
    r2 = r2_score(y_test_inv, y_pred_inv)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    # Predict next day's closing price
    last_30 = df.drop(columns='Close').values[-30:]  
    last_30 = np.expand_dims(last_30, axis=0)       
    next_scaled = model.predict(last_30)

    dummy_next = np.zeros((1, df.shape[1]))
    dummy_next[0, close_idx] = next_scaled[0][0]
    next_day_price = scaler.inverse_transform(dummy_next)[0][close_idx]

    #print(f"\033[92mNext predicted closing price for {ticker}: ₹{next_day_price:.2f}\033[0m")
    return y_test_inv, y_pred_inv, rmse, r2, next_day_price


    # Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_inv, label='Actual Close Price')
    # plt.plot(y_pred_inv, label='Predicted Close Price')
    # plt.title(f'{ticker} GRU: Actual vs Predicted Close Prices')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
def plot_actual_vs_predicted(y_true, y_pred, title='GRU Prediction'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
def train_and_save_model(ticker):

    df=pd.read_csv(f"df_for_gru_scaled_{ticker}.csv")
    df.dropna(inplace=True)
    df=df.astype('float32')
    
    X, y = create_sequence(df, target_col='Close', seq_len=30)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = Sequential()
    model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mse')
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    model.save(f"gru_model_{ticker}.keras")
    with open(f"last_trained_{ticker}.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

