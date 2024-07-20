if __name__ == '__main__':
    import torch.utils
    import torch
    from torch import nn
    #? Data
    import requests
    import zipfile
    from pathlib import Path
    import os
    #? vizualizace - fotek
    from PIL import Image
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision import datasets
    #! DtaLoader
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #? připaví cestu pro data
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    #? pokud slozku pro data uz nemáme vytvoří se
    if image_path.is_dir():
        print(f"{image_path} data uz máš... přeskakuji stahování")
    else:
        print(f"{image_path} data neexistují ... stahuji")
        image_path.mkdir(parents=True,exist_ok=True)

    #? stahování dat
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading...")
        f.write(request.content)

        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r") as zip_ref:
            print("Odzipování")
            zip_ref.extractall(image_path)


    #! DATA - pokus porozumnět jim

    def walk_through_dir(dir_path):
        """ Projde dir_path a vrátí to co obsahuje"""
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"Tady je {len(dirnames)} složek a {len(filenames)} fotek v '{dirpath}'.")

    #! Rozdělení dat na traing a testing části
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    """
    #! Pro vizualizaci
    #? získáme cety daných fotek
    image_path_list = list(image_path.glob("*/*/*.jpg")) #? glob dá všechny cety dohromady * - cokoly
    #? vybereme náhodnou fotku pomocí random.choice()
    random_image_path = random.choice(image_path_list)
    #? získání class fotky
    image_class = random_image_path.parent.stem #? stem = class dané fotky
    """
    #! Model TinyVGG
    #todo Postup:
    #_ 1. Transformace

    #! Transformace dat
    simple_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    train_data_simple = datasets.ImageFolder(root=train_dir,
                                      transform=simple_transform,
                                      target_transform=None)
    test_data_simple = datasets.ImageFolder(root=test_dir,
                                     transform=simple_transform)

    BATCH_SIZE = 1                             #! originalně 32
    WORKERS= os.cpu_count()
    train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                  batch_size=BATCH_SIZE,
                                  num_workers=WORKERS,
                                  shuffle=True)
    test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                 batch_size=BATCH_SIZE,
                                 num_workers=WORKERS,
                                 shuffle=False)
    class_names =train_data_simple.classes
    class_dict = train_data_simple.class_to_idx
    #! Baseline model
    #! Architecture: conv62 - relu -> conv60 - relu60 - maxpool30 -> conv28 - relu28 -> conv26 - relu26 -maxpool13

    class TinyModelCustom(nn.Module):
        def __init__(self, input_shape:int,hidden_units:int,output_shape:int):
            """
            Kopie modelu TinyVGG 
            """
            super().__init__()
            self.block1=nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU()
            )
            self.block2=nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)
            )
            self.block3=nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU()
            )
            self.block4=nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)
            )
            self.classifier=nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*13*13,
                        out_features=output_shape)
            )
        def forward(self,x):
            x = self.block1(x)
            #print(f"Shape of x: {x.shape}")
            x = self.block2(x)
            #print(f"Shape of x: {x.shape}")
            x = self.block3(x)
            #print(f"Shape of x: {x.shape}")
            x = self.block4(x)
            #print(f"Shape of x: {x.shape}")
            x = self.classifier(x)
            return x
    model_0 = TinyModelCustom(input_shape=3,
                    hidden_units=10,
                    output_shape=len(class_names)).to(device)
    image_batch,label_batch = next(iter(train_dataloader_simple))
    #print(model_0(image_batch))

    #! Vizualizace - tabulka
    """
        from torchinfo import summary
    summary(model=model_0,
            input_size=(BATCH_SIZE,3,64,64))
    """

    
    #!Trainig


    def train_loop(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device=device):
        #! stejný popis pro test_s
        #? Vyhodnocovací hodnoty: loss,acc
        model.train()
        train_loss,train_acc = 0,0
        #!Loop
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            #? Forward pass -> vytvoření predikcí -> logitů
            y_pred=model(X)
            #? Loss
            loss = loss_fn(y_pred,y)
            #? Celková loss
            train_loss +=loss.item()
           #? Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #? Acc
            y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
            #? Celková acc
            train_acc +=(y_pred_class==y).sum().item()/len(y_pred) #? celkový počet správných predikcí / počet predikcí
            #if batch%400==0:
            #    print(f"Looked at: {batch*len(X)}/{len(data_loader.dataset)} samples")
        #? Průmerné hodnoty pro loss a acc
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_loop(model:torch.nn.Module,
                  dataloader:torch.utils.data.DataLoader,
                  loss_fn:torch.nn.Module,
                  device:torch.device = device):
        
        model.eval()
        test_acc,test_loss =0,0
        with torch.inference_mode():
            for batch,(X,y) in enumerate(dataloader):
                X,y = X.to(device), y.to(device)
                test_pred =model(X)
                loss = loss_fn(test_pred,y)
                test_loss += loss.item()
                test_pred_class = torch.argmax(torch.softmax(test_pred,dim=1), dim=1)
                test_acc += (test_pred_class==y).sum().item()/len(test_pred)
        test_loss /= len(dataloader)
        test_acc/=len(dataloader)
                #print(f"\nTest loss: {test_loss} | Test acc: {test_acc}%")
        return test_loss, test_acc
    #! Inicializace treniku

    from tqdm.auto import tqdm
    def train(model:torch.nn.Module,
              train_data_loader:torch.utils.data.DataLoader,
              test_data_loader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              # accuracy_fn
              device:torch.device=device,
              epochs: int =5):
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}
        for epoch in tqdm(range(epochs)):
            print(f" Epoch: {epoch+1}\n================")
            train_loss, train_acc =train_loop(model=model,
                                        dataloader=train_data_loader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
            test_loss,test_acc = test_loop(model=model,
                                        dataloader= test_data_loader,
                                        loss_fn=loss_fn,
                                        device=device)
            print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            #? Update results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        return results
    torch.manual_seed(42)                         #! *******Epochy**********
    NUM_EPOCHS = 3                             

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                                lr=0.001)         #! *******LR**********
    from timeit import default_timer as timer
    start_time = timer()
    vysledek_0 = train(model=model_0,
                     epochs=NUM_EPOCHS,
                     train_data_loader=train_dataloader_simple,
                     test_data_loader=test_dataloader_simple,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     device=device)
    end_time = timer()
    print(f"Total trainig time: {end_time-start_time:.3f} seconds")
    
#! Vizualizace -> pomocí grafu - loss curve
#? result keys
    vysledek_0.keys()
    def plot_loss_curves(results:dict[str,list[float]]):
        """Zobrazí křivku loss funkce"""
        #? získání dat
        loss = results["train_loss"]
        test_loss = results["test_loss"]
        accuracy = results["train_acc"]
        test_accuracy = results["test_acc"]
        epochs=range(len(results["train_loss"]))

        #? zobrazení
        plt.figure(figsize=(15,7))

        #? loss
        plt.subplot(1,2,1)
        plt.plot(epochs,loss,label="train_loss")
        plt.plot(epochs,test_loss,label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        #? acc
        plt.subplot(1,2,2)
        plt.plot(epochs,accuracy,label="train_accuracy")
        plt.plot(epochs,test_accuracy,label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
    plot_loss_curves(vysledek_0)
    #! přečíst
    # https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic
    import pandas as pd
    model_0_df = pd.DataFrame(vysledek_0)
    print(model_0_df)

    directory_path = Path("data/model_df/")
    filename = "model_0_df"
    
    full_path = os.path.join(directory_path,filename)
    model_0_df.to_csv(full_path, index=False)

