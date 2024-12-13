import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

# Load XRP data from Yahoo Finance (replace with your preferred API if needed)
@st.cache
def load_data():
    data = yf.download('XRP-AUD', start='2018-01-01', end='2034-12-31')
    return data

# Preprocess data for LSTM model
def preprocess_data(data):
    # Use 'Close' prices
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create dataset for training LSTM (use previous 60 days to predict the next day)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be compatible with LSTM input (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    return model

# Predict and visualize results
def predict(model, data, scaler):
    # Prepare the last 60 days for prediction
    last_60_days = data[-60:][['Close']].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.reshape(last_60_days_scaled, (1, 60, 1))
    
    # Make prediction
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

# Streamlit App UI
st.title("XRP Price Prediction")

# Load data and preprocess
data = load_data()
X_train, y_train, scaler = preprocess_data(data)

# Define model path
model_path = 'xrp_model.h5'

# Load or train model
if os.path.exists(model_path):
    model = load_model(model_path)  # Load pre-trained model
else:
    model = build_model()
    model = train_model(model, X_train, y_train)
    model.save('my_model.keras')  # Save model for future use

# Predict the next day's price
predicted_price = predict(model, data, scaler)
st.write(f"Predicted Next Day XRP Price: {predicted_price:.2f} USD")

# Plot historical data and predictions
st.subheader("Historical Data")
st.line_chart(data['Close'])

# Predict for test data (use the last 60 days of the dataset as the test set)
test_data = data[-100:]  # Taking a small portion for test
X_test, y_test, _ = preprocess_data(test_data)

# Predict the entire test set
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Flatten the arrays to ensure they are 1-dimensional
predicted_prices = predicted_prices.flatten()
actual_prices = test_data['Close'][60:].values

# Create a DataFrame with the predicted and actual values
result_df = pd.DataFrame({
    'Predicted': predicted_prices,
    'Actual': actual_prices
})

# Ensure the DataFrame columns are 1-dimensional
st.subheader("Predicted vs Actual Prices")
st.line_chart(result_df)
