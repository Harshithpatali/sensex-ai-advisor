from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_KEY")

def generate_insight(ticker, predicted_price, sentiment_score):
    prompt = f"""
Stock: {ticker}
Predicted price: {predicted_price:.2f}
Recent news sentiment: {sentiment_score:.2f}

Generate a concise investment insight for a retail investor.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
