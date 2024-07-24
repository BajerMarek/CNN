
if __name__ == '__main__':
    # 1. Get data
    import os
    import requests
    import zipfile
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
      print(f"{image_path} directory exists.")
    else: 
      print(f"Did not find {image_path} directory, creating...")
      image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data (images from GitHub)
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
      request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
      print("Downloading pizza, steak, sushi data...")
      f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
      print(f"Unzipping pizza, steak, suhsi data to {image_path}")
      zip_ref.extractall(image_path) 

    train_dir = image_path / "train"
    test_dir = image_path / "test"
    # Write transform for turning images into tensors
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()])

    # Use ImageFolder to create dataset(s)
    from torchvision import datasets
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images 
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform,
                                     target_transform=None)
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    # Turn train and test Datasets into DataLoaders
    from torch.utils.data import DataLoader
    BATCH_SIZE = 1
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=os.cpu_count(),
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=os.cpu_count(),
                                 shuffle=False)

    import torch
    from torch import nn

    class TinyVGG(nn.Module):
      def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape))

      def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Layer 1 shape: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Layer 2 shape: {x.shape}")
        x = self.classifier(x)
        # print(f"Layer 3 shape: {x.shape}")
        return x
    
    model_0 = TinyVGG(input_shape = 3,
                      hidden_units=10,
                      output_shape=len(class_names)).to(device)

    def train_step(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer):
    
      # Put the model in train mode
      model.train()

      # Setup train loss and train accuracy values
      train_loss, train_acc = 0, 0

      # Loop through data loader and data batches
      for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device) 

        # 1. Forward pass
        y_pred = model(X)
        # print(y_pred)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad 
        optimizer.zero_grad()

        # 4. Loss backward 
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumualte accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

      # Adjust metrics to get average loss and average accuracy per batch
      train_loss = train_loss / len(dataloader)
      train_acc = train_acc / len(dataloader)
      return train_loss, train_acc 
      
    def test_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module):
    
      # Put model in eval mode
      model.eval()

      # Setup the test loss and test accuracy values
      test_loss, test_acc = 0, 0

      # Turn on inference context manager
      with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)
          # print(test_pred_logits)

          # 2. Calculuate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

      # Adjust metrics to get average loss and accuracy per batch
      test_loss = test_loss / len(dataloader)
      test_acc = test_acc / len(dataloader)
      return test_loss, test_acc
    from tqdm.auto import tqdm

    def train(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
              epochs: int = 5):
    
      # Create results dictionary
      results = {"train_loss": [],
                 "train_acc": [],
                 "test_loss": [],
                 "test_acc": []}

      # Loop through the training and testing steps for a number of epochs
      for epoch in tqdm(range(epochs)):
        # Train step
        train_loss, train_acc = train_step(model=model, 
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        # Test step
        test_loss, test_acc = test_step(model=model, 
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # Print out what's happening
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}"
        )

        # Update the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

      # Return the results dictionary
      return results


    # Train for 5 epochs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_0 = TinyVGG(input_shape=3,
                      hidden_units=10,
                      output_shape=len(class_names)).to(device)

    loss_fn=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)
    from timeit import default_timer as timer
    start_time = timer()

    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            epochs=5)
    print(model_0_results)
    end_time = timer()
    print(f"Total time: {(end_time - start_time)/60}")