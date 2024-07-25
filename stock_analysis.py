import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stock symbol and the period for data
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetch data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Create a new column 'Previous Close' to use as a feature
data['Previous Close'] = data['Close'].shift(1)

# Drop rows with NaN values (resulting from the shift operation)
data = data.dropna()

# Define features and target variable
X = data[['Previous Close']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Prices', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
