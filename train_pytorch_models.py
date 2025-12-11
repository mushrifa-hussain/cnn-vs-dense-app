import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device
device = torch.device("cpu")

# MNIST transforms
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST
train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ---------------------------
# Dense Model (MLP)
# ---------------------------
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

dense_model = DenseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dense_model.parameters(), lr=0.001)

print("Training Dense Model...")
for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        output = dense_model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(dense_model.state_dict(), "dense_model.pt")
print("Dense model saved.")

# ---------------------------
# CNN Model
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

cnn_model = CNN().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("Training CNN Model...")
for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        output = cnn_model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(cnn_model.state_dict(), "cnn_model.pt")
print("CNN model saved.")
