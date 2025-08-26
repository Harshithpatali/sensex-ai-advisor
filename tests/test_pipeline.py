# tests/test_pipeline.py
from src.data_loader import fetch_stock_data
from src.news_sentiment import fetch_news
from src.preprocessing import merge_stock_sentiment, scale_features
from src.model import create_dataset, build_lstm
from src.generate_insights import generate_insight

def test_pipeline():
    ticker = "^BSESN"
    df_stock = fetch_stock_data(ticker)
    df_news = fetch_news(ticker, num_articles=2, api_key="YOUR_SERPAPI_KEY")
    
    df_features = merge_stock_sentiment(df_stock, df_news)
    df_scaled, _ = scale_features(df_features, ["Open","High","Low","Close","Volume","sentiment"])
    
    X, y = create_dataset(df_scaled, ["Open","High","Low","Close","Volume","sentiment"], time_steps=10)
    model = build_lstm((X.shape[1], X.shape[2]))
    pred = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    
    insight = generate_insight(ticker, pred[0][0], df_news["sentiment"].mean())
    
    # Assertions to confirm everything worked
    assert df_stock.shape[0] > 0
    assert df_news.shape[0] > 0
    assert X.shape[0] > 0
    assert pred[0][0] > 0
    assert isinstance(insight, str)
