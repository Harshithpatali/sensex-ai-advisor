import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date="2023-01-01", end_date="2025-12-31"):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

if __name__ == "__main__":
    df_sensex = fetch_stock_data("^BSESN")
    df_sensex.to_csv("../data/raw/sensex.csv", index=False)
    print(df_sensex.head())
