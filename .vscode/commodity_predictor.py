import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
Commodity_Ticker = "GC=F"
Start_Date = "2010-01-01"
End_Date = "2023-10-31"
def fetch_and_prepare_data(ticker):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=Start_Date, end=End_Date)
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df = df.dropna()
    df["Prediction"] = df["Close"].shift(-1)
    df = df.dropna()
    return df
def train_model(df):
    features = ["Open","High","Low","Close","Volume","MA_10","MA_50"]
    X = df[features]
    y = df["Prediction"]
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, shuffle = False)
    print("Training Random Forest Model...")
    model = RandomForestRegressor(n_estimators= 100, random_state= 42)
    model.fit(X_train,y_train)

    return model, X_test, y_test
def evaluate_and_plot(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,predictions)

    print(f"\nModel Performance: ")
    print(f"Root Mean Squared Error: {rmse: .2f}")
    print(f"R^2 Score: {r2:.2f}")

    plt.figure(figsize=(14,7))
    plt.plot(y_test.index, y_test, label="Actual Price", color="blue", alpha= 0.6)
    plt.plot(y_test.index, predictions, label="Predicted Price", color= "Orange", alpha = 0.7)
    plt.title("Gold Price Prediction: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (usd)")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    print("--- Commodity Price Prediction APP ---")

    data = fetch_and_prepare_data(Commodity_Ticker)
    print(f"Data processed: {len(data)} records.")

    rf_model, X_test, y_test = train_model(data)

    evaluate_and_plot(rf_model, X_test, y_test)
    print("Process Complete.")