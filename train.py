import torch
import torch.nn as nn
from torch.optim import Adam
from models import TransformerTranslator
from dataset import load_datasets

def train_model(train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTranslator().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item() * input_ids.size(0)
                
                _, predicted = torch.max(outputs, 2)
                correct += (predicted == labels).sum().item()
                total += labels.size(0) * labels.size(1)
        
        val_loss /= len(val_loader.dataset)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    return model
