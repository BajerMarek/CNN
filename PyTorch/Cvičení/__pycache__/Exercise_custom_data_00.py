if __name__ == '__main__':
  # Import torch
  import torch
  from torch import nn
  from pathlib import Path
  import requests
  import zipfile
  import matplotlib.pyplot as plt
  import random
  import torch.utils
  import torch.utils.data
  import torchvision
  from torchvision import transforms, datasets
  from PIL import Image
  # Exercises require PyTorch > 1.10.0
  #print(torch.__version__)       #? zobrazí verzy pytorch

  # Setup device agnostic code
  device = "cuda" if torch.cuda.is_available() else "cpu"
  #print(device)

  device = "cuda" if torch.cuda.is_available() else "cpu"

  data_path = Path("data/")
  image_path = data_path / "Cviceni_custom_00"

  if image_path.is_dir():
      print(f"{image_path} data jsou uz stažena -> nestahuji")
  else:
      print(f"{image_path} data nejsou stažena -> stahuji")
      image_path.mkdir(parents=True,exist_ok=True)        #? udělá složku

      with open(data_path / "pizza_steak_sushi.zip", "wb") as f:  #? otevře sloýku na dané místš
          request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")    #? stáhne zip z githubu
          print("Downloading...")
          f.write(request.content)    #? vloží zip do složky

      
      with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r") as zip_ref:   #? otevře zip na daném místě
          print("Odzipování")
          zip_ref.extractall(image_path)      #? extrahuje zip 

  # 2. Become one with the data
  import os
  def walk_through_dir(dir_path):
    """Walks through dir_path returning file counts of its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
      print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

  #walk_through_dir(image_path)   #? vypíše informace o dané kmenové struktuře -> o daní složce a všech složkách vní

  # Setup train and testing paths
  train_dir = image_path / "train"    #? udělá složku train
  test_dir = image_path / "test"      #? udělá složku test
  # Visualize an image
  image_path_list = list(image_path.glob("*/*/*.jpg"))    #? udělá list včech cest k fotkám
  print(image_path_list[0])   
  random_image_path = random.choice(image_path_list)      #? vybere náhodnou cestu z listu fotek
  random_image_tensor = torchvision.io.read_image(str(random_image_path)) #? převede fotku na danné cestě na tensor
  # Do the image visualization with matplotlib
  plt.imshow(random_image_tensor.permute(1,2,0))      #? zobrazí forku kterná má otočené dimenze aby to bylo správně pro matplotlib  
  plt.show()                                          #? [heigt, width, color channels] -> po otočení


  # 3.1 Transforming data with torchvision.transforms
  transform = transforms.Compose([transforms.TrivialAugmentWide(num_magnitude_bins=31),   #? přemnění PIL fotku podle zadani
                                  transforms.Resize(size=(64,64)),                        #? umenší fotku
                                  transforms.ToTensor()])                                 #? převede na tensor
                                                                                          #? trivial augment -> sada ůprav
  # Write transform for turning images into tensors
  #train_dataset = torch.utils.data.Dataset(train_dir)
  #test_dataset = torch.utils.data.Dataset(test_dir)

  # Write a function to plot transformed images
  def zobraz_trans_fotku(image_path:str,
                        transform):
    """Zobrazí fotku sdanou path a transformovanou daným transformem"""
    image = Image.open(image_path)           #? převede fotku s danou path na PIL fotku
    transformed_image = transform(image)     #? použije transform
    
    plt.imshow(transformed_image.permute(1,2,0)) #? zobrazí fotku
    plt.show()

  zobraz_trans_fotku(str(random_image_path),transform=transform)

  train_data = datasets.ImageFolder(root=train_dir,       #? převede na datset
                                    transform=transform,
                                    target_transform=None)
  test_data = datasets.ImageFolder(root=test_dir,
                                  transform=transform,
                                  target_transform=None)
  class_names = train_data.classes
  class_dict = train_data.class_to_idx
  print(class_names)
  print(class_dict)
  print(f"Train: {len(train_data)} | Test: {len(test_data)}")

  BATCH_SIZE = 32                     #? počet fotek které se budou zpracovávat naráz
  NUM_WORKERS = os.cpu_count()        #? počet jader cpu které budou použity na zpracování dat
  train_dataloader = torch.utils.data.DataLoader(dataset=train_data,          #? vytvoří dataloader -> format fotek vhodný pro zpracování
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS,
                                                shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS,
                                                shuffle=False)

  class TinyVggKlon(nn.Module):
    """Bude se jednat o model napodubující TinyVgg z webu CNN explainer """
    def __init__(self,hidden_units,inpute_shape,output_shape):
        super().__init__()
        self.block1 = nn.Sequential(              #? Skládá jednotlivé vrstvy dohromady
          nn.Conv2d(in_channels=inpute_shape,    #? vytvoří convolutionalní vrstvu o dvou dimenzích
                    out_channels=hidden_units,
                    kernel_size=3,
                    padding=1,
                    stride=1),
          nn.ReLU(),                              #? prožene data z předešlé vrstvy funkcí ReLU
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    padding=1,
                    stride=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)             #! nevím dolpnit
        )
        self.block2 = nn.Sequential(
              nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    padding=1,
                    stride=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        padding=1,
                        stride=1),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2)
          )
        self.Classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*16*16,
                    out_features=output_shape)
        )
    def forward(self,x):         #? prožene data jednotlivými vrstvami
        x = self.block1(x)
      #print(f"Shape of x: {x.shape}")
        x = self.block2(x)
      #print(f"Shape of x: {x.shape}")
        x = self.Classifier(x)
        return x

  def train_step(model: torch.nn.Module,    #? vytváří loop ve kterém se model učí
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer):
    
    # Put the model in train mode
    model.train()   #? nastaví model do módu pro učení -> z maximalizuje se účinost učení

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader and data batches
    for batch, (X,y) in enumerate(dataloader):  #? opakuj pro počet dat v data loaderu
      # Send data to target device
      X,y = X.to(device),y.to(device)                 #? převední na device
      # 1. Forward pass
      model_pred = model(X)                           #? vytvoření predikcí
      # 2. Calculate and accumulate loss
      loss = loss_fn(model_pred,y)                    #? vytvoření loss
      train_loss += loss.item()
      # 3. Optimizer zero grad 
      optimizer.zero_grad()
      # 4. Loss backward 
      loss.backward()
      # 5. Optimizer step
      optimizer.step()
      # Calculate and accumualte accuracy metric across all batches
      model_pred_class = torch.argmax(torch.softmax(model_pred,dim=1), dim=1)
      #? Celková acc
      train_acc += (model_pred_class == y).sum().item()/len(model_pred) #? celkový počet správných predikcí / počet predikcí

    # Adjust metrics to get average loss and average accuracy per batch
    train_loss = train_loss/ len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

  def test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module):
    
    # Put the model in train mode
    model.eval()

    # Setup train loss and train accuracy values
    test_loss, test_acc = 0, 0

    # Loop through data loader and data batches
    with torch.inference_mode():
      for batch, (X,y) in enumerate(dataloader):
        # Send data to target device
        X,y = X.to(device),y.to(device)
        # 1. Forward pass
        model_pred = model(X)
        # 2. Calculate and accumulate loss
        loss = loss_fn(model_pred,y)
        test_loss += loss.item()
        # Calculate and accumualte accuracy metric across all batches
        model_pred_labels = model_pred.argmax(dim=1)
        #? Celková acc
        test_acc += (model_pred_labels == y).sum().item()/len(model_pred) #? celkový počet správných predikcí / počet predikcí

    # Adjust metrics to get average loss and average accuracy per batch
    test_loss = test_loss/ len(dataloader)
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
  model_cviceni = TinyVggKlon(hidden_units=10,
                              inpute_shape=3,
                              output_shape=len(class_names)).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model_cviceni.parameters(),
                              lr = 0.001)        #! ********************* LR **************************

  NUM_EPOCHS = 5
  from timeit import default_timer as timer

  start_time = timer()
  vysledek = train(model=model_cviceni,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  epochs=NUM_EPOCHS)

  print(vysledek)
  end_time = timer()
  print(f"Total time: {(end_time - start_time)/60}")

  import pandas as pd
  model_cviceni_df = pd.DataFrame(vysledek)
  print(model_cviceni_df)

  directory_path = Path("data/model_df/")
  filename = "model_cviceni_df"
  print("-----------------------")
  full_path = os.path.join(directory_path,filename)
  model_cviceni_df.to_csv(full_path, index=False)
  print("-----------------------")