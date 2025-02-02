# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from utils import load_data
from torch.nn.functional import softmax

def train_model(config, train_loader, test_loader):
    
    # Initialize the model
    model = get_model(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training settings
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    learning_rate = config['train']['learning_rate']
    weight_decay = config['train']['weight_decay']
    criterion_ = config['train']['criterion']
    optimizer_ = config['train']['optimizer']
    
    # Define the loss function and optimizer
    
    criterion = nn.CrossEntropyLoss() if criterion_ == 'CrossEntropyLoss' else nn.BCEWithLogitsLoss() if criterion_ == 'BCEWithLogitsLoss' else nn.MSELoss() if criterion_ =='MSELoss' else None
    if optimizer_ == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_ == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_ == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_ == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_}")
    
    print("Model Summary:", config['model'])
    print("criterion: ", criterion, "optimizer: ", optimizer)
    print(f"Training {config['model']['model_name']} for {epochs} epochs with batch size {batch_size}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.long()  # Ensure proper types

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        #evaluate_model(model, test_loader) #uncomment to evaluate after each epoch
    print("Finished Training")
    return model



def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    device = next(model.parameters()).device

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.long()  # Ensure proper types
            outputs = model(inputs)  # Forward pass
            predictions = torch.argmax(outputs, dim=1)  # Get predicted class
            correct += (predictions == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total samples

    accuracy = correct / total
    # print(label_sum/len(test_loader))
    # print(labels)
    print(f"Test Accuracy: {accuracy:.2%}")
    return accuracy

