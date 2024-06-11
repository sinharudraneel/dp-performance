import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

# Define the single-layer neural network
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate input data and target data
X = torch.randn(100, 3).to(device)
y = torch.randn(100, 1).to(device)

# Set input size and output size
input_size = 3
output_size = 1

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X, y)
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, criterion, and optimizer
model = SingleLayerNN(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Test the model with a new sample
with torch.no_grad():
    test_sample = torch.randn(1, 3).to(device)
    prediction = model(test_sample)
    print(f'Prediction for test sample: {prediction.item():.4f}')
