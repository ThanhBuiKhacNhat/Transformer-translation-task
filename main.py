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

# Load data
data = pd.read_csv("opus_books_en_hu.csv")

# Split data
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
model = TransformerTranslator(num_layers=3, d_model=256, num_heads=4, hidden_dim=256, 
                               input_vocab_size=tokenizer.vocab_size, target_vocab_size=tokenizer.vocab_size,
                               max_seq_len=128, dropout=0.1).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Train the model
train_losses, val_losses, val_accuracies = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

# Evaluate the model
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

# Save the trained model parameters
torch.save(model.state_dict(), "transformer_model.pth")

# Save metrics to CSV
metrics = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv('metrics.csv', index=False)
