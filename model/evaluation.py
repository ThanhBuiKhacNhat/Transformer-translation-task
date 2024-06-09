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

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, target_ids, input_mask, target_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            test_loss += loss.item() * input_ids.size(0)

            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) * labels.size(1)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    # Print test loss and accuracy
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return test_loss, accuracy
