# src/news_sentiment.py

import pandas as pd
import re
from serpapi import GoogleSearch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r"\@\w+|\#", "", text)  # remove @mentions and hashtags
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# -----------------------------
# Fetch news from SerpAPI
# -----------------------------
def fetch_news(ticker, num_articles=10, api_key="YOUR_SERPAPI_KEY"):
    """
    Fetch latest news for a ticker/index using SerpAPI.
    Returns a DataFrame with columns: title, snippet, link, cleaned_text, sentiment
    """
    params = {
        "engine": "google_news",
        "q": ticker,
        "api_key": api_key,
        "num": num_articles
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    # If no news results, return empty DataFrame
    news_list = results.get("news_results", [])
    if not news_list:
        return pd.DataFrame(columns=["title", "snippet", "link", "cleaned_text", "sentiment"])

    # Convert to DataFrame safely
    df = pd.DataFrame(news_list)

    # Ensure required columns exist
    for col in ["title", "snippet", "link"]:
        if col not in df.columns:
            df[col] = ""

    # Clean text for sentiment analysis
    df["cleaned_text"] = (df["title"].fillna("") + " " + df["snippet"].fillna("")).apply(clean_text)

    return df

# -----------------------------
# Sentiment Analysis
# -----------------------------
def analyze_sentiment(df):
    """
    Uses HuggingFace transformers to get sentiment for each news article.
    """
    # Load pretrained sentiment model
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Apply sentiment analysis
    sentiments = sentiment_pipeline(df["cleaned_text"].tolist())
    df["sentiment"] = [s["score"] if "score" in s else 0.0 for s in sentiments]

    return df

# -----------------------------
# Full pipeline
# -----------------------------
def get_news_sentiment(ticker, num_articles=10, api_key="5e9b133801fb14a3f8a628d8569d42c65baf4c0220a4f030387d749010edfa66"):
    df_news = fetch_news(ticker, num_articles, api_key)
    if df_news.empty:
        return df_news
    df_news = analyze_sentiment(df_news)
    return df_news
