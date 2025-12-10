import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import time

# ------------------------------------------------------
# 1. Load CSV files
# ------------------------------------------------------
prices = pd.read_csv("./price/TSLA.csv")
news   = pd.read_csv("./sorted_filtered/TSLA_news_sorted.csv")

# ------------------------------------------------------
# 2. Normalize date formats
# ------------------------------------------------------
prices['date'] = pd.to_datetime(prices['date']).dt.date
news['Date']   = pd.to_datetime(news['Date']).dt.date

# ------------------------------------------------------
# 3. Sentiment model
# ------------------------------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

def get_score(text):
    try:
        label = sentiment_pipeline(text[:512])[0]["label"]
        return {"LABEL_2": 1, "LABEL_1": 0, "LABEL_0": -1}.get(label, 0)
    except:
        return 0

# ------------------------------------------------------
# 4. Safe translator
# ------------------------------------------------------

# ------------------------------------------------------
# 5. OUTPUT CSV
# ------------------------------------------------------
output_path = "./price/TSLA_with_sentiment.csv"
header_written = False

# Process only dates > 2018
filtered_dates = [d for d in sorted(prices['date']) if d.year > 2015]

# ------------------------------------------------------
# 6. Process with progress bar
# ------------------------------------------------------
with open(output_path, "w") as f:

    for d in tqdm(filtered_dates, desc="Processing dates > 2015", unit="date"):
        # print(d)

        price_row = prices[prices['date'] == d].copy()

        day_news = news[news['Date'] == d]
        news_count = len(day_news)
        news_flag = 1 if news_count > 0 else 0

        date_scores = []
        if news_count > 0:
            for _, row in day_news.iterrows():

                # -----------------------------------------
                # 1. Select article text or fallback to title
                # -----------------------------------------
                article = row["Article"]

                if not article or len(str(article).strip()) < 10:
                    article = row["Article_title"]

                if not article or len(str(article).strip()) < 3:
                    continue

                score = get_score(article)
                date_scores.append(score)

        # -----------------------------------------
        # 4. Compute average sentiment
        # -----------------------------------------
        avg_sentiment = sum(date_scores) / len(date_scores) if date_scores else 0

        # if date_scores:
        #     print("Scores for", d, ":", date_scores,avg_sentiment)

        # -----------------------------------------
        # 5. Add new features to price row
        # -----------------------------------------
        price_row["news_flag"] = news_flag
        price_row["news_count"] = news_count
        price_row["avg_sentiment"] = avg_sentiment

        # -----------------------------------------
        # 6. Write enriched row to final CSV
        # -----------------------------------------
        if not header_written:
            price_row.to_csv(f, index=False, header=True)
            header_written = True
        else:
            price_row.to_csv(f, index=False, header=False)

print("\nCSV successfully created at:", output_path)
