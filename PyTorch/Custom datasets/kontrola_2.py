import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
def download_and_extract_data():
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating...")
        image_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)
    
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print(f"Unzipping pizza, steak, sushi data to {image_path}")
        zip_ref.extractall(image_path)
    
    return image_path

def get_data_loaders(image_path, batch_size=32, num_workers=4):
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])
    
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_dataloader, test_dataloader, train_data.classes

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred_class = y_pred.argmax(dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs=5):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = download_and_extract_data()
    train_dataloader, test_dataloader, class_names = get_data_loaders(image_path, batch_size=32, num_workers=os.cpu_count())
    
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    from timeit import default_timer as timer
    start_time = timer()
    
    results = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs=5)
    
    end_time = timer()
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
    
    def plot_loss_curves(results):
        epochs = range(len(results["train_loss"]))
        
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, results["train_loss"], label="train_loss")
        plt.plot(epochs, results["test_loss"], label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, results["train_acc"], label="train_accuracy")
        plt.plot(epochs, results["test_acc"], label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        
        plt.show()
    
    plot_loss_curves(results)