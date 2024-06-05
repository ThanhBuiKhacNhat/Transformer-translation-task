from dataset import load_datasets
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    # Load datasets
    train_loader, val_loader, test_loader = load_datasets("test.csv")

    # Train model
    model = train_model(train_loader, val_loader)

    # Evaluate model
    evaluate_model(model, test_loader)
