# My Project

This project is a stock price prediction system that applies advanced time series analysis, technical indicator engineering, and deep learning to forecast the next day's closing stock price. The application supports predictions for six selected Indian stocks and presents results via a professional Streamlit dashboard.

## Supported Stocks

The application currently supports prediction for the following six stocks:

- RELIANCE (Reliance Industries Limited)
- TCS (Tata Consultancy Services)
- COLPAL (Colgate-Palmolive India)
- INFY (Infosys Limited)
- PAGEIND (Page Industries Limited)
- ITC (ITC Limited)

## Overview

The modeling pipeline begins with exploratory data analysis, including stationarity and seasonality checks, followed by the computation of several technical indicators. These features are then used to train a GRU (Gated Recurrent Unit) neural network for time series forecasting.

The system is designed with modularity in mind — from preprocessing to prediction — and is packaged in a user-friendly interface.

## Core Components

### 1. Exploratory Data Analysis (EDA)
- Stationarity checks using Augmented Dickey-Fuller (ADF) test
- Seasonal decomposition and trend inspection
- Volatility visualization

### 2. Technical Indicator Engineering
The following indicators are computed and used as features:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Bollinger Bands
- Stochastic Oscillator

### 3. GRU-Based Forecasting Model
- Sequence-to-one deep learning model using GRU layers
- Multivariate input combining historical price and technical indicators
- Trained to predict the next day's closing price

### 4. Streamlit Application
- Supports prediction for 6 major Indian stocks
- Clean dark-themed dashboard
- Includes historical visualization and model predictions

## How to Run

1. Run the preprocessing script to prepare the data for the selected stock tickers:

```bash
python preprocessing.py


2. Launch the streamlit app
streamlit run app.py

