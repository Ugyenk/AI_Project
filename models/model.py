import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json

# Load and prepare dataset 
X1 = np.load("data/temp/X_cropped.npy")
y1 = np.load("data/temp/y_cropped.npy")

data_dir = "data/processed_data"
classes = ["0", "1"]
X2, y2 = [], []

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((47, 64)),
    transforms.ToTensor(),
])

for label, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("L")
            tensor = transform(img)
            X2.append(tensor.numpy())
            y2.append(label)

X2 = np.stack(X2)
y2 = np.array(y2)

X1 = X1 / 255.0
X1 = np.transpose(X1, (0, 3, 1, 2))
X_combined = np.concatenate([X1, X2], axis=0)
y_combined = np.concatenate([y1, y2], axis=0)

X_tensor = torch.tensor(X_combined, dtype=torch.float32)
y_tensor = torch.tensor(y_combined, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Models same as before
class DinoNet1(nn.Module):
    def __init__(self):
        super(DinoNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 10 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)

class DinoNet2(nn.Module):
    def __init__(self):
        super(DinoNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 10 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class DinoNet3(nn.Module):
    def __init__(self):
        super(DinoNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7)
        self.pool2 = nn.MaxPool2d(2, 2)
        self._to_linear = None
        self._get_flattened_size()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 47, 64)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Training function
def train_model(model, dataloader, model_name, epochs=10, lr=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        print(f"{model_name} Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")

    save_path = f"{model_name}_lr{lr}_epochs{epochs}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")

    return history

# Hyperparameter tuning 
def hyperparameter_tuning():
    learning_rates = [1e-3, 1e-4, 1e-5]  # 3 learning rates
    batch_size = 32
    epochs = 10

    model_classes = {
        "DinoNet1": DinoNet1,
        "DinoNet2": DinoNet2,
        "DinoNet3": DinoNet3,
    }

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_name, model_class in model_classes.items():
        
        results[model_name] = {}
        for lr in learning_rates:
            print(f"\nTraining {model_name} with lr={lr} and batch_size={batch_size}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model = model_class()
            history = train_model(model, dataloader, f"{model_name}_lr{lr}_bs{batch_size}", epochs=epochs, lr=lr, device=device)
            results[model_name][f"lr_{lr}_bs_{batch_size}"] = history

    with open("hyperparameter_tuning_results.json", "w") as f:
        json.dump(results, f)

    print("\nHyperparameter tuning complete! Results saved in 'hyperparameter_tuning_results.json'.")

if __name__ == "__main__":
    hyperparameter_tuning()
