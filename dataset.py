import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        en_sentence = self.data.iloc[idx]['en']
        hu_sentence = self.data.iloc[idx]['hu']

        encoding = self.tokenizer(en_sentence, hu_sentence, 
                                  return_tensors='pt', 
                                  max_length=self.max_length, 
                                  padding='max_length', 
                                  truncation=True)
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Add labels
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # Mask token
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def load_datasets(file_path, test_size=0.1, val_size=0.1, max_length=128, batch_size=8):
    data = pd.read_csv(file_path)
    
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_dataset = TranslationDataset(train_data, tokenizer, max_length)
    val_dataset = TranslationDataset(val_data, tokenizer, max_length)
    test_dataset = TranslationDataset(test_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
