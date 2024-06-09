import torch


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        tuple: Test loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            # Move input and label tensors to the specified device
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            test_loss += loss.item() * input_ids.size(0)  # Accumulate test loss

            # Compute predictions
            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).sum().item()  # Accumulate number of correct predictions
            total += labels.size(0) * labels.size(1)  # Accumulate total number of labels

    # Compute average test loss
    test_loss /= len(test_loader.dataset)
    # Compute accuracy
    accuracy = correct / total

    # Print test loss and accuracy
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return test_loss, accuracy
