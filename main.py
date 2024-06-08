import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from dataset import TranslationDataset
from models import TransformerTranslator
from train import train
from evaluate import evaluate
import warnings
from transformers import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", message="overflowing tokens are not returned")
logging.set_verbosity_error()

# Load data
data = pd.read_csv("opus_books_en_hu.csv")

# Split data into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Initialize datasets
train_dataset = TranslationDataset(train_data, tokenizer)
val_dataset = TranslationDataset(val_data, tokenizer)
test_dataset = TranslationDataset(test_data, tokenizer)

# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerTranslator(
    num_layers=6, 
    d_model=256, 
    num_heads=8, 
    hidden_dim=512, 
    input_vocab_size=tokenizer.vocab_size, 
    target_vocab_size=tokenizer.vocab_size,
    max_seq_len=128, 
    dropout=0.1
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_losses, val_losses, val_accuracies = train(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device
)

# Evaluate the model
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

# Save the trained model parameters
torch.save(model.state_dict(), "transformer_model.pth")

# Initialize an empty DataFrame
metrics_df = pd.DataFrame(columns=['test_loss', 'test_accuracy'])

# Assume you have a loop over epochs
for epoch in range(20):
    # Append metrics to DataFrame
    metrics_df = metrics_df.append({
        'test_loss': test_loss, 
        'test_accuracy': test_accuracy
    }, ignore_index=True)

# Save DataFrame to CSV
metrics_df.to_csv('metrics.csv', index=False)
