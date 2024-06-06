import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TranslationDataset(Dataset):
    """Custom Dataset for translation tasks using BERT tokenizer."""
    
    def __init__(self, data, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data.
            tokenizer (transformers.PreTrainedTokenizer): BERT tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        # Get the English and Hungarian sentences
        en_sentence = self.data.iloc[idx]['en']
        hu_sentence = self.data.iloc[idx]['hu']

        # Tokenize the sentences
        encoding = self.tokenizer(en_sentence, hu_sentence, 
                                  return_tensors='pt', 
                                  max_length=self.max_length, 
                                  padding='max_length', 
                                  truncation=True)
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Add labels: copy input_ids and mask padding tokens
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # Ignore padding token in loss calculation
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


