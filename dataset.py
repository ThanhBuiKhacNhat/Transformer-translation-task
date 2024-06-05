import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
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

def load_datasets(filepath):
    data = pd.read_csv(filepath)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)
    test_dataset = TranslationDataset(test_data, tokenizer)
    return train_dataset, val_dataset, test_dataset
