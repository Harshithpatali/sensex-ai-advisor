from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_loader import fetch_stock_data
from src.news_sentiment import fetch_news
from src.preprocessing import merge_stock_sentiment, scale_features
from src.model import create_dataset, build_lstm
from src.generate_insights import generate_insight

def pipeline_task():
    ticker = "^BSESN"
    df_stock = fetch_stock_data(ticker)
    df_news = fetch_news(ticker, num_articles=10, api_key="YOUR_SERPAPI_KEY")
    
    df_features = merge_stock_sentiment(df_stock, df_news)
    df_features, _ = scale_features(df_features, ["Open","High","Low","Close","Volume","sentiment"])
    
    X, y = create_dataset(df_features, ["Open","High","Low","Close","Volume","sentiment"], time_steps=10)
    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=32)
    
    predicted_price = model.predict(X[-1].reshape(1,X.shape[1],X.shape[2]))[0][0]
    avg_sentiment = df_news["sentiment"].mean()
    insight = generate_insight(ticker, predicted_price, avg_sentiment)
    
    df_features.to_csv("data/processed/sensex_prediction.csv", index=False)

dag = DAG(
    'sensex_ai_pipeline',
    start_date=datetime(2025, 8, 26),
    schedule_interval='@daily',
    catchup=False
)

run_pipeline = PythonOperator(
    task_id='run_pipeline',
    python_callable=pipeline_task,
    dag=dag
)
