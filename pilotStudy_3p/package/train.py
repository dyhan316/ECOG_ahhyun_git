# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from utils import load_data
from torch.nn.functional import softmax
import numpy as np
import copy

def train_model(config, train_loader, valid_loader):
    
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
    
    early_stop = EarlyStoppingWithModelCopy(patience=10, verbose=True)
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
        
        #validation loop
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.float(), labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_vloss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {running_loss / len(train_loader):.4f}, Avg Val Loss: {running_vloss / len(valid_loader):.4f}",
              f"Training Acc : {evaluate_model(model, train_loader):.2%}, Validation Acc : {evaluate_model(model, valid_loader):.2%}")
        
        #early stopping
        early_stop(running_vloss, model)
        if early_stop.early_stop:
            print("Early stopping")
            early_stop.load_best_weights(model)
            break
        
        # evaluate_model(model, valid_loader) #uncomment to evaluate after each epoch
    print("Finished Training")
    return model



def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    device = next(model.parameters()).device
    model.eval()
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



class EarlyStoppingWithModelCopy() : 
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        if self.best_model_weights is not None:
            print(f"Loading best model weights" )
            model.load_state_dict(self.best_model_weights)
        else:
            raise ValueError("No best model weights to load.")