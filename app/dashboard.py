# app/dashboard.py

import sys
import os

# Add project root so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
from src.data_loader import fetch_stock_data
from src.preprocessing import merge_stock_sentiment, scale_features
from src.model import create_dataset, build_lstm
from src.generate_insights import generate_insight
from src.news_sentiment import get_news_sentiment

st.set_page_config(page_title="AI Financial Advisor - Sensex Edition", layout="wide")

st.title("AI-Powered Financial Advisor - SENSEX Edition")

# -----------------------------
# Sidebar - Select Stock/Index
# -----------------------------
ticker = st.selectbox(
    "Select Stock/Index",
    ["^BSESN"]  # Add more tickers if needed
)

num_articles = st.slider("Number of News Articles to Analyze", min_value=5, max_value=20, value=10)

api_key = st.text_input("SerpAPI Key", type="password")

if st.button("Fetch Data"):

    if not api_key:
        st.warning("Please enter your SerpAPI key to fetch news.")
        st.stop()

    # -----------------------------
    # Fetch stock data
    # -----------------------------
    st.info(f"Fetching data for {ticker}...")
    df_stock = fetch_stock_data(ticker)

    if df_stock.empty:
        st.error("No stock data found!")
        st.stop()
    else:
        st.success("Stock data fetched!")
        st.dataframe(df_stock.tail(5))

    # -----------------------------
    # Fetch news & sentiment
    # -----------------------------
    st.info("Fetching news and performing sentiment analysis...")
    df_news = get_news_sentiment(ticker, num_articles=num_articles, api_key=api_key)

    if df_news.empty:
        st.warning("No news data found!")
    else:
        st.success("News and sentiment fetched!")
        st.dataframe(df_news[["title", "snippet", "sentiment"]].head())

    # -----------------------------
    # Merge stock + news
    # -----------------------------
    st.info("Preparing features...")
    df_features = merge_stock_sentiment(df_stock, df_news)
    df_scaled, scaler = scale_features(df_features, ["Open", "High", "Low", "Close", "Volume", "sentiment"])
    st.success("Features ready!")

    # -----------------------------
    # Prepare dataset for LSTM
    # -----------------------------
    X, y = create_dataset(df_scaled, ["Open", "High", "Low", "Close", "Volume", "sentiment"], time_steps=10)

    # -----------------------------
    # Build & predict with LSTM
    # -----------------------------
    model = build_lstm((X.shape[1], X.shape[2]))
    st.info("Training LSTM model (1 epoch for demo)...")
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    pred_price = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0][0]
    st.success(f"Predicted Price: {pred_price:.2f}")

    # -----------------------------
    # Generate insights
    # -----------------------------
    avg_sentiment = df_news["sentiment"].mean() if not df_news.empty else 0
    insight_text = generate_insight(ticker, pred_price, avg_sentiment)

    st.info("AI-Generated Insights:")
    st.write(insight_text)
