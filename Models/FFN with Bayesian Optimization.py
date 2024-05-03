# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class CostSensitiveLoss(torch.nn.Module):
    def __init__(self, cost_matrix):
        super(CostSensitiveLoss, self).__init__()
        self.cost_matrix = cost_matrix

    def forward(self, outputs, labels):
        # Compute the standard cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Compute the cost-sensitive loss
        weights = torch.Tensor([self.cost_matrix[label.item()] for label in labels])
        cs_loss = torch.nn.functional.cross_entropy(outputs, labels, weight=weights)

        return cs_loss


# Define the model class
class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Define the training loop
def train(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss= criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(dataloader)}, Loss: {loss.item()}")

# Generate some example data

trainf = pd.read_csv("train_data_1 (2).csv")
train_size = int(len(trainf))

train_inputs = trainf.drop('HeartDisease',axis=1)
train_targets = trainf['HeartDisease']

# convert the inputs and targets to NumPy arrays
train_input_array = train_inputs.values
train_input_array = train_input_array.astype('float32')
train_target_array = train_targets.values
train_target_array = train_target_array.astype('float32')

# convert the NumPy arrays to PyTorch tensors
train_inputs_tensor = torch.tensor(train_input_array)
train_targets_tensor = torch.tensor(train_target_array)
train_targets_tensor = train_targets_tensor.reshape(-1, 1)

valf = pd.read_csv("test_data_1.csv")

val_inputs = valf.drop('HeartDisease',axis=1)
val_targets = valf['HeartDisease']

# convert the inputs and targets to NumPy arrays
val_inputs_array = val_inputs.values
val_inputs_array = val_inputs_array.astype('float32')
val_targets_array = val_targets.values
val_targets_array = val_targets_array.astype('float32')

# convert the NumPy arrays to PyTorch tensors
val_inputs_tensor = torch.tensor(val_inputs_array)
val_targets_tensor = torch.tensor(val_targets_array)
val_targets_tensor = val_targets_tensor.reshape(-1, 1)
#inputs = torch.randn(1000, 10)
#targets = torch.randint(low=0, high=2, size=(1000, 1)).float()

# Split the data into a training and validation set
#train_size = int(0.8 * len(inputs))
#train_inputs, train_targets = inputs[:train_size], targets[:train_size]
#val_inputs, val_targets = inputs[train_size:], targets[train_size:]

# Define the hyperparameters
input_size = 38
output_size = 1
num_epochs = 300  
batch_size = 32  # 128
cost_matrix = ([0, 3], [1, 0])

# Train the model with different hidden sizes
for hidden_size in [9, 10, 11]:
    print(f"Training with hidden size = {hidden_size}")
    model = BPNN(input_size, hidden_size, output_size)
 #   optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = CostSensitiveLoss(cost_matrix)
    
    train_data = TensorDataset(train_inputs_tensor, train_targets_tensor)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train(model, train_dataloader, optimizer, criterion, num_epochs)
    
    val_data = TensorDataset(val_inputs_tensor, val_targets_tensor)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        model.eval()
        val_outputs = model(val_inputs_tensor)
        val_loss = criterion(val_outputs, val_targets_tensor)
        val_accuracy = ((val_outputs > 0.5).float() == val_targets_tensor).float().mean()
        print(f"Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}")

