import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def merge_stock_sentiment(stock_csv, news_csv):
    df_stock = pd.read_csv(stock_csv, parse_dates=["Date"])
    df_news = pd.read_csv(news_csv, parse_dates=["date"])
    
    daily_sentiment = df_news.groupby("date")["sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"date": "Date"}, inplace=True)
    
    df = pd.merge(df_stock, daily_sentiment, on="Date", how="left")
    df["sentiment"].fillna(0, inplace=True)
    return df

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

if __name__ == "__main__":
    df = merge_stock_sentiment("../data/raw/sensex.csv", "../data/processed/sensex_news_sentiment.csv")
    df, scaler = scale_features(df, ["Open", "High", "Low", "Close", "Volume", "sentiment"])
    df.to_csv("../data/processed/sensex_features.csv", index=False)
    print(df.head())
