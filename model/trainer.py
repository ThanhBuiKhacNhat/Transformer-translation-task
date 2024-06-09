import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, target_ids, input_mask, target_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                target_mask = batch['target_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, target_ids, input_mask, target_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item() * input_ids.size(0)

                _, predicted = torch.max(outputs, 2)
                correct += (predicted == labels).sum().item()
                total += labels.size(0) * labels.size(1)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    return train_losses, val_losses, val_accuracies
