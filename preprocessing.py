import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

def fetch_stock_data(ticker,years = 5):
    end_date=dt.datetime.now()
    start_date = end_date - dt.timedelta(days=years * 365)
    df=yf.download(ticker, start=start_date,end=end_date,auto_adjust=True)
    return df


def add_indicators(df,ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df= df.copy()
    #to take only price :
    close_price = df[f'Close_{ticker}']
    high_price =df[f'High_{ticker}']
    low_price= df[f'Low_{ticker}']
    volume= df[f'Volume_{ticker}']
    df['sma7']=SMAIndicator(close=close_price,window=7).sma_indicator()
    df['sma21']=SMAIndicator(close=close_price, window=21).sma_indicator()
    df['ema30']=EMAIndicator(close=close_price, window=30).ema_indicator()

    #MACD
    macd = MACD(close=close_price)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_diff']= macd.macd_diff()
    #RSI
    df['rsi']= RSIIndicator(close=close_price).rsi()

    #  Bollinger Bands
    bollinger = BollingerBands(close=close_price)
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    #  Stochastic Oscillator
    stoch = StochasticOscillator(high=high_price, low=low_price, close=close_price)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    #Volume Indicators
    df['obv'] = OnBalanceVolumeIndicator(close=close_price, volume= volume).on_balance_volume()
    df['vpt'] = VolumePriceTrendIndicator(close=close_price, volume= volume).volume_price_trend()
    # Price-based features
    df['return_1d'] = close_price.pct_change(1)
    df['return_5d'] = close_price.pct_change(5)
    df['return_10d'] = close_price.pct_change(10)
    
    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
    df.rename(columns={f'Close_{ticker}': 'Close'}, inplace=True)

    
    return df

def select_features_with_lasso(df , target_column='Close'):
    df_clean = df.dropna()
    X=df_clean.drop(columns = [target_column])
    y=df_clean[target_column]
    X_train , X_test,y_train,y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=5,random_state=0,max_iter=10000)
    lasso.fit(X_train_scaled, y_train.values.ravel())
    selected = pd.Series(lasso.coef_,index =X.columns)
    selected = selected[selected != 0]
    return list(selected.index) + [target_column]

def scale_and_save_data_for_gru(df, selected_columns , out_csv_path , scaler_path):
    df_filtered = df[selected_columns].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filtered)

    joblib.dump(scaler, scaler_path)

    df_scaled = pd.DataFrame(scaled, columns=selected_columns)
    df_scaled.to_csv(out_csv_path, index=False)
    return df_scaled
def preprocess_pipeline_for_gru(ticker):
    df = fetch_stock_data(ticker)
    df = add_indicators(df,ticker)
    selected = select_features_with_lasso(df)
    df_scaled = scale_and_save_data_for_gru(
        df,
        selected,
        out_csv_path=f"df_for_gru_scaled_{ticker.replace('.NS','')}.csv",
        scaler_path=f"scaler_{ticker.replace('.NS','')}.save"
    )
    return df_scaled


#preprocess_pipeline_for_gru("RELIANCE.NS")
preprocess_pipeline_for_gru("TCS.NS")

#preprocess_pipeline_for_gru("PAGEIND.NS")
#preprocess_pipeline_for_gru("INFY.NS")
#preprocess_pipeline_for_gru("COLPAL.NS")
#preprocess_pipeline_for_gru("ITC.NS")