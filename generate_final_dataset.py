import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import time


prices = pd.read_csv("./price/MSFT.csv")
news   = pd.read_csv("./sorted_filtered/MSFT_news_sorted.csv")

prices['date'] = pd.to_datetime(prices['date']).dt.date
news['Date']   = pd.to_datetime(news['Date']).dt.date

finbert_model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",  # alias for text-classification
    model=model,
    tokenizer=tokenizer
)

def get_score(text: str) -> int:
    """
    Map FinBERT output to numerical score:
      positive -> +1
      neutral  ->  0
      negative -> -1
    """
    try:
        # Truncate very long texts a bit before tokenization
        text = str(text)
        result = sentiment_pipeline(text[:512])[0]
        label = result["label"].lower()  # e.g. 'positive', 'negative', 'neutral'

        mapping = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }
        return mapping.get(label, 0)
    except Exception:
        return 0

output_path = "./price/MSFT_with_sentiment.csv"
header_written = False

# Process only dates > 2015 (your current condition)
filtered_dates = [d for d in sorted(prices['date']) if d.year > 2015]

with open(output_path, "w") as f:

    for d in tqdm(filtered_dates, desc="Processing dates > 2022", unit="date"):
        price_row = prices[prices['date'] == d].copy()

        day_news = news[news['Date'] == d]
        news_count = len(day_news)
        news_flag = 1 if news_count > 0 else 0

        date_scores = []
        if news_count > 0:
            for _, row in day_news.iterrows():

                article = row.get("Article", "")
               

                if not article or len(str(article).strip()) < 10:
                    article = row.get("Article_title", "")

                if not article or len(str(article).strip()) < 3:
                    continue

                score = get_score(article)
                print(article)
                date_scores.append(score)

        avg_sentiment = sum(date_scores) / len(date_scores) if date_scores else 0
        price_row["news_flag"] = news_flag
        price_row["news_count"] = news_count
        price_row["avg_sentiment"] = avg_sentiment

        if not header_written:
            price_row.to_csv(f, index=False, header=True)
            header_written = True
        else:
            price_row.to_csv(f, index=False, header=False)

print("\nCSV successfully created at:", output_path)
