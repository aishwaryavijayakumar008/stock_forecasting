import pandas as pd
import os


news_path = "All_external.csv"
df_news = pd.read_csv(news_path)
df_news["Article_title"] = df_news["Article_title"].astype(str)


magnificent_7 = {
    "AAPL": ["Apple", "AAPL"],
    "MSFT": ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "AMZN": ["Amazon", "AMZN"],
    "NVDA": ["Nvidia", "NVDA"],
    "META": ["Meta", "Facebook", "META"],
    "TSLA": ["Tesla", "TSLA"]
}

print("Starting filtering...")

# Ensure output folders exist
filtered_folder = "filtered"
sorted_folder = "sorted_filtered"
os.makedirs(filtered_folder, exist_ok=True)
os.makedirs(sorted_folder, exist_ok=True)


for ticker, keywords in magnificent_7.items():
    pattern = "|".join(keywords)
    df_filtered = df_news[df_news["Article_title"].str.contains(pattern, case=False, na=False)]
    output_path = os.path.join(filtered_folder, f"{ticker}_news.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved filtered file: {output_path}")


csv_files = [f for f in os.listdir(filtered_folder) if f.endswith(".csv")]

print("Sorting filtered files...")

for file in csv_files:
    file_path = os.path.join(filtered_folder, file)
    df = pd.read_csv(file_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(by="Date")

    sorted_path = os.path.join(sorted_folder, file.replace(".csv", "_sorted.csv"))
    df.to_csv(sorted_path, index=False)
    print(f"Saved sorted file: {sorted_path}")

print("Completed pipeline.")
