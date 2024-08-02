if __name__ == '__main__':
    import os
    import requests
    import zipfile
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torch import nn
    import matplotlib.pyplot as plt
    import wandb
    cislo_pokusu = 5
    wandb.init(project="Tracking_00",
               name=f"RUN-{cislo_pokusu}",
               
               config={"NUM_EPOCHS":10,
                       "BATCH_SIZE":32,
                       "lr":0.001})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(device)
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
        
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    transform_trivial = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()])

    #! Datasets, dataloaders
    train_data = datasets.ImageFolder(root=train_dir,
                                              transform=transform_trivial,
                                              target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                             transform=transform_trivial,
                                             target_transform=None)
    class_names =train_data.classes
    class_dict = train_data.class_to_idx
    BATCH_SIZE = wandb.config["BATCH_SIZE"]
    train_dataloader = DataLoader(dataset=train_data,
                                          batch_size=BATCH_SIZE,
                                          num_workers=os.cpu_count(),
                                          shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                         batch_size=BATCH_SIZE,
                                         num_workers=os.cpu_count(),
                                         shuffle=False)
    class TinyModelCustom(nn.Module):
        def __init__(self, input_shape,hidden_units,output_shape):
            super().__init__()
            self.block1=nn.Sequential(
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
            self.block2=nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier=nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*16*16,
                        out_features=output_shape)
            )
        def forward(self,x):
            x = self.block1(x)
            #print(f"Shape of x: {x.shape}")
            x = self.block2(x)
            #print(f"Shape of x: {x.shape}")
            x = self.classifier(x)
            return x

                                                            #!Trainig
    def train_loop(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer):
        #! stejný popis pro test_s
        #? Vyhodnocovací hodnoty: loss,acc
        model.train()
        train_loss,train_acc = 0,0
        #!Loop
        for batch,(X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
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
            train_acc += (y_pred_class == y).sum().item()/len(y_pred) #? celkový počet správných predikcí / počet predikcí

            #if batch%400==0:
            #    print(f"Looked at: {batch*len(X)}/{len(data_loader.dataset)} samples")
        #? Průmerné hodnoty pro loss a acc
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        return train_loss, train_acc

    def test_loop(model:torch.nn.Module,
                  dataloader:torch.utils.data.DataLoader,
                  loss_fn:torch.nn.Module):
        
        model.eval()
        test_acc,test_loss =0,0
        with torch.inference_mode():
            for batch,(X,y) in enumerate(dataloader):
                X,y = X.to(device), y.to(device)
                test_pred_logits =model(X)
                loss = loss_fn(test_pred_logits,y)
                test_loss += loss.item()
                test_pred_labels =test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_logits))

        test_loss =test_loss / len(dataloader)
        test_acc=test_acc / len(dataloader)

                #print(f"\nTest loss: {test_loss} | Test acc: {test_acc}%")
        return test_loss, test_acc
    #! Inicializace treniku

    from tqdm.auto import tqdm
    def train(model:torch.nn.Module,
              train_dataloader:torch.utils.data.DataLoader,
              test_dataloader:torch.utils.data.DataLoader,
              optimizer:torch.optim.Optimizer,
              loss_fn:torch.nn.Module =nn.CrossEntropyLoss(),
              epochs: int =5):
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc =train_loop(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
            test_loss,test_acc = test_loop(model=model,
                                        dataloader= test_dataloader,
                                        loss_fn=loss_fn)
            
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
            wandb.log({"train_loss":train_loss,
                       "train_acc":train_acc,
                       "test_loss":test_loss,
                       "test_acc":test_acc})
        return results
    

    model_01 = TinyModelCustom(input_shape=3,
                          hidden_units=20,
                          output_shape=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_01.parameters(),  #! jiny optimizer nez model0
                               lr=wandb.config["lr"])
                                                                                        #! sledování
    wandb.watch(models=model_01,
                criterion=loss_fn,
                log_graph=True,
                log="all",
                log_freq=1)

    NUM_EPOCHS = wandb.config["NUM_EPOCHS"]
    from timeit import default_timer as timer
    start_time = timer()

    vysledek_01 =train(model=model_01,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
          epochs=NUM_EPOCHS)
    img,label = next(iter(train_dataloader))
    print(vysledek_01)
    end_time = timer()
    wandb.log({"Total_time_[s]":end_time - start_time})
    print(f"Total time: {(end_time - start_time)/60}")
    
    import pandas as pd
    model_02_df = pd.DataFrame(vysledek_01)
    print(model_02_df)

    directory_path = Path("data/model_df/")
    filename = "model_02_df"
    
    full_path = os.path.join(directory_path,filename)
    model_02_df.to_csv(full_path, index=False)

