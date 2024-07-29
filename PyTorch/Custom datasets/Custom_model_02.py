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
    BATCH_SIZE = 16
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

    """
    
    #! Pro vizualizaci
    #? získáme cety daných fotek
    image_path_list = list(image_path.glob("*/*/*.jpg")) #? glob dá všechny cety dohromady * - cokoly
    #? vybereme náhodnou fotku pomocí random.choice()
    random_image_path = random.choice(image_path_list)
    #? získání class fotky
    image_class = random_image_path.parent.stem #? stem = class dané fotky
    """

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
        return results
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_01 = TinyModelCustom(input_shape=3,
                          hidden_units=10,
                          output_shape=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_01.parameters(),  #! jiny optimizer nez model0
                               lr=0.001)

<<<<<<< Updated upstream
    NUM_EPOCHS = 5
=======
    NUM_EPOCHS = 1
>>>>>>> Stashed changes
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
    print(f"Total time: {(end_time - start_time)/60}")
    
    import pandas as pd
    model_02_df = pd.DataFrame(vysledek_01)
    print(model_02_df)

    directory_path = Path("data/model_df/")
    filename = "model_02_df"
    
    full_path = os.path.join(directory_path,filename)
    model_02_df.to_csv(full_path, index=False)

    #! Vizualizace -> pomocí grafu - loss curve
    #? result keys
    vysledek_01.keys()
    def plot_loss_curves(results:dict[str,list[float]]):
        
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
    plot_loss_curves(vysledek_01)   
    #! Cíl ?
    #todo Stáhnutí jedné specialní fotky a provedení predikce na ní
    import requests

    custom_image_path = data_path/"04-pizza-dad.jpeg"
    if not custom_image_path.is_file():
        with open(custom_image_path,"wb") as f:
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Stahování {custom_image_path}...")
            f.write(request.content)    #? stáhne tu forku
    else:
        print(f"{custom_image_path} je už stažená, přeskakuji stahování") 

    #? převedení na správný formát -> stejný jako data ne kterých trénovala
    #? + správný device
    
    #? přečtení fotky
    import torchvision
    custom_img_uint8 = torchvision.io.read_image(str(custom_image_path))
    #plt.imshow(custom_img_uint8.permute(1,2,0))
    #plt.show()
    print(f"Custom image tensor:\n {custom_img_uint8}")
    print(f"Custom image shape:\n {custom_img_uint8.shape}")
    print(f"Custom image datatype:\n {custom_img_uint8.dtype}")
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) /255    #? standartně jsou hodnoty RGB v rozmezí 0 -255
                                                                                                #?  -> výsledek mezi 1 a 0
    print(custom_image)


    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) /255    #? standartně jsou hodnoty RGB v rozmezí 0 -255 + zmnění na tensor
                                                                                                 #?  -> výsledek mezi 1 a 0
    custom_image_transformed = transforms.Compose([
        transforms.Resize(size=(64,64))
        ])
    custom_image_transformed = custom_image_transformed(custom_image)
    print(f"Originalní shape: {custom_img_uint8.shape}")
    print(f"Transformed shape: {custom_image_transformed.shape}")
    #! Vizualizace zobrazované fotky po transformaci
    plt.imshow(custom_image_transformed.permute(1,2,0))
    plt.show()
    model_01.eval()
    with torch.inference_mode():
        custom_image_pred = model_01(custom_image_transformed.unsqueeze(dim=0).to(device))
        print(custom_image_pred)

    #! Co sem udělali :
    #? převedení fotky na tensor
    #? převedení na správný data typ (float32)
    #? převední na správný shape (64,64)

    #! Funkcionalizace + zobrazení výspedků

    #? pravděpodobnost správného výspedku
    custom_image_pred_probs = torch.softmax(custom_image_pred,dim=1)
    print(custom_image_pred_probs)

    #? přeměna pravděpoděpodobnosti na název
    custom_image_pred_labels = torch.argmax(custom_image_pred_probs,dim=1)
    print(custom_image_pred_labels)
    print(class_names[custom_image_pred_labels])

    #! Funkce
    #todo Idelní výsledek:
    #_ Dáme naší funkci image_path a ona nám zobrazí danou fotku + naní provede predikci

    def zobraz_fotku(image_path:str):
        #? příprava fotky
        image = torchvision.io.read_image(str(image_path)).type(torch.float32)/255
        transformed_image = transforms.Compose([transforms.Resize(size=(64,64))])
        transformed_image = transformed_image(image)
        #? predikce na fotce
        model_01.eval()
        with torch.inference_mode():
            image_pred = model_01(transformed_image.unsqueeze(dim=0).to(device))
        #? Dekodacce predikce
        image_pred_probs = torch.softmax(image_pred,dim=1)
        image_pred_labels = torch.argmax(image_pred_probs)
        #!Vizualizace
        image_vizualizace = torchvision.io.read_image(str(image_path)) 
        plt.imshow(image_vizualizace.permute(1,2,0))
        plt.title(f"Jnémo jídla je: {class_names[image_pred_labels]}")
        plt.show()
    zobraz_fotku(custom_image_path)
    #!Nefunguje -> Cast error details: Unable to cast Python instance of type <class 'pathlib.WindowsPath'> to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
"""

    def zobraz_fotku_2(model:torch.nn.Module,
                 image_path:str,
                 class_names: list[str] = None,
                 transform = None,
                 device=device):
        #? příprava fotky
        image= torchvision.io.read_image(image_path).type(torch.float32)/255
        if transform:
            image = transform(image)
        #? predikce na fotce
        model.to(device)
        model.eval()
        with torch.inference_mode():
            image_pred = model(image.unsqueeze(dim=0).to(device))   #? přídá dimenzi pro batche v nulté dimezi -> .unsqueeze(dim=0)
        #? Dekodacce predikce
        image_pred_probs = torch.softmax(image_pred,dim=1)
        image_pred_labels = torch.argmax(image_pred_probs)
        #!Vizualizace
        image_vizualizace = torchvision.io.read_image(image_path) 
        plt.imshow(image_vizualizace.permute(1,2,0))
        if class_names:
            plt.title(f"Jnémo jídla je(2): {class_names[image_pred_labels]} s pravděpodobností: {image_pred_probs.max():.3f}")
        else:
            plt.title(f"Predikce: {image_pred_labels} | Pravděpoobnost: {image_pred_probs.max():.3f}")
        plt.axis(False)
        plt.show()
    transform = transforms.Resize(size=(64,64))
    zobraz_fotku_2(model=model_01,
                 image_path=custom_image_path,
                 transform=transform,
                 device=device,
                 class_names=class_names)
       """ 