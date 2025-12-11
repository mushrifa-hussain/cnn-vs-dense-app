import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

st.title("CNN vs Dense Network Comparison App")
st.write("This app compares the performance of a Dense Network and a Convolutional Neural Network on MNIST images.")

device = torch.device("cpu")

# Define Dense model
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

# Define CNN model
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

# Load models
dense_model = DenseNet()
dense_model.load_state_dict(torch.load("dense_model.pt", map_location=device))
dense_model.eval()

cnn_model = CNN()
cnn_model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
cnn_model.eval()

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Evaluate accuracy
def evaluate(model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total

dense_acc = evaluate(dense_model)
cnn_acc = evaluate(cnn_model)

st.subheader("Accuracy Comparison")
st.write(f"**Dense Network Accuracy:** {dense_acc:.4f}")
st.write(f"**CNN Accuracy:** {cnn_acc:.4f}")

fig, ax = plt.subplots()
ax.bar(["Dense", "CNN"], [dense_acc, cnn_acc], color=["red", "green"])
ax.set_ylim(0.8, 1.0)
st.pyplot(fig)

# Sample predictions
st.subheader("Sample Predictions")
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
indices = np.random.choice(len(test_dataset), 5, replace=False)

for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].axis("off")

    dense_pred = torch.argmax(dense_model(img.unsqueeze(0))).item()
    cnn_pred = torch.argmax(cnn_model(img.unsqueeze(0))).item()

    axes[i].set_title(f"T:{label}\nD:{dense_pred} C:{cnn_pred}", fontsize=8)

st.pyplot(fig)

