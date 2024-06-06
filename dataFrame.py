import pandas as pd
from datasets import load_dataset

# Tải tập dữ liệu từ Hugging Face Hub
dataset = load_dataset("opus_books", "en-hu")

# Chuyển đổi tập dữ liệu thành DataFrame
data = dataset['train'][:]
df = pd.DataFrame(data)

# Tách cột 'translation' thành hai cột riêng biệt 'en' và 'hu'
df = df.join(pd.json_normalize(df['translation']))
df.drop(columns=['translation'], inplace=True)

# Lưu DataFrame thành file CSV
df.to_csv("opus_books_en_hu.csv", index=False)

print("Dữ liệu đã được lưu vào file opus_books_en_hu.csv")


