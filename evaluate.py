import torch
from torch.nn import CrossEntropyLoss

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            test_loss += loss.item() * input_ids.size(0)
            
            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) * labels.size(1)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
