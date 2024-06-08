import pandas as pd
from datasets import load_dataset

# Load dataset from Hugging Face Hub
dataset = load_dataset("opus_books", "en-hu")

# Convert dataset to DataFrame
data = dataset['train'][:]
df = pd.DataFrame(data)

# Split the 'translation' column into separate 'en' and 'hu' columns
df = df.join(pd.json_normalize(df['translation']))
df.drop(columns=['translation'], inplace=True)

# Save DataFrame to a CSV file
df.to_csv("opus_books_en_hu.csv", index=False)

print("Data has been saved to the opus_books_en_hu.csv file")
