# Import repositories and packages 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the stock symbol and the period for data
print('Market Vision')

# User inputs
stock_symbol = str(input('Input Stock Symbol Ex ("AAPL"): '))
start_date = input('Input Start Date Ex (2020-01-01): ')

# Use the current date as the end date
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Check if data is empty
if data.empty:
    print(f"No data available for the stock symbol {stock_symbol}. Please check the symbol and date range.")
else:
    # Proceed with processing the data
    print("First few rows of the dataset:")
    print(data.head())

    # Create a new column 'Previous Close' to use as a feature
    data['Previous Close'] = data['Close'].shift(1)

    # Drop rows with NaN values (resulting from the shift operation)
    data = data.dropna()

    # Define features and target variable
    X = data[['Previous Close']]
    y = data['Close']

    # Initialize and train the model on all available data
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions for the next 7 days
    last_close_price = data['Close'].iloc[-1]
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
    future_predictions = []

    for _ in range(7):
        predicted_price = model.predict([[last_close_price]])[0]
        future_predictions.append(predicted_price)
        last_close_price = predicted_price

    # Print future predictions
    print("Predicted prices for the next 7 days:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: ${price:.2f}")

    # Plot actual and predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Prices', color='blue')
    plt.plot(future_dates, future_predictions, label='Predicted Prices', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction for the Next Week')
    plt.legend()
    plt.show()
