# Computer vision project
#! Torch
import torch
from torch import nn 
#! Torchvision
import torchvision 
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#! Matplotlib
import matplotlib.pyplot as plt
#! Další
import requests
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm 
#! Data set
#? Fasion mnist z torch vision
#? trainig data

train_data = datasets.FashionMNIST(
    root="data",                                      #? kam se stáhnou
    train=True,                                       #? Tranigový dataset ano / ne
    download=True,                                    #? Stáhnout ano / ne
    transform=torchvision.transforms.ToTensor(),      #? Transformaovat ano / ne / jak
    target_transform=None                             #? Jak transformovat labels / target
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
#! Vizualizace dat
image, label = train_data[0]
print(len(train_data), len(test_data))
class_names = train_data.classes
print(class_names)
class_to_idx = train_data.class_to_idx
print(class_to_idx)
print(f"Image shape: {image.shape} -> color_chanel, height, width")
print(f"Image label: {class_names[label]}")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.show()

#? nahodné fotky
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0,len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows,cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()

#! Data loader
#_ Zmnění data na opakovatélná Pythonem
#_ Přesněji vytvoří mini dávky dat -> Pro efektivnější funkcy modelu
#_ Učí se z 32 brázků najednou a né z 16 000 obrázků
#_ Zároveň to dává síti možnost 
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
print(f"DataLoudery: {train_dataloader, test_dataloader}")
print(f"Délka train_dataloaderu: {len(train_dataloader)}, batches o {BATCH_SIZE}...")
print(f"Délka test_dataloader: {len(test_dataloader)}, batches o {BATCH_SIZE}...")

#? Vlastnosti train_dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)
#? Zobrazení ukazky z batch
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch),size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()
print(f"Image size: {img.shape}")
print(f"Lable: {label}, label size: {label.shape}")

#! Základní model
#_ V pžípadě experimentování s problémém je vhodné vytvořit zákaldní model
#_ který se bude upravovat a vylopšovat na základě pokusu
#_ základní model je vpodstatě nejjednoduší model - kostra modelu
#_ Vpodstatě začít jednoduše a postupně přidávát složitost

#? Flatten layer
flatten_model = nn.Flatten()                    #! převede height a width na vektor
#? jeden vzorek
x = train_features_batch[0]
print(x.shape, x)
#? Flaten the sampel
output = flatten_model(x)   #? forward pass
#? Co se stalo
print(f"Shape před flattenig: {x.shape} -> [color_chanels,  height, width]")
print(f"Shape after flattenig: {output.shape} -> [colorchanels, height * width]")
print(output.squeeze())

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=784,          #? 28 * 28
                              hidden_units=10,          #? pocet neuronů 
                              output_shape=len(class_names))

dummy_x = torch.rand([1,1,28,28])
print(model_0(dummy_x))
print(model_0.state_dict())

#todo Optimizer a loss funkce
#_ loss funkce - vzheledm k tomu že se jedná o multiclass problem tedy pouzijeme nn.CrossEntropyLoss
#_ Otimizer - torch.optim.SGD() (stochastic gradiant descent)
#_ Evaluation metrice - protože se jedná o problém s klasifikací použijeme přesost pro hodnocení

#! Download hepler functos z githubu
if Path("helper_functions.py").is_file():
    print("heplper_functions.py jsou již stažené -> přesakuji stahování")
else:
    print("Stahuji hepler_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
#? inportovaní z helper functions
from helper_functions import accuracy_fn

#! Loss funkce  a optimizer
loss_fn= nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.1)

#! Funkce na časování našeho experimentu
#todo Ecperimentace
#_ Často se hlídá u modelu:
#_výkonost a rychlost

def print_train_time(start: float,
                     end: float,
                     device: torch.device =None):
    """ Vypíše rozdíl mezi časem začátku a koncem"""
    total_time = end -start
    print(f"Train time on {device}: {total_time} sekund")
    return total_time

#start_time = timer()
#? nějáký kód
#end_time =timer()
#print_train_time(start=start_time,end=end_time, device="cpu")

#! Trainig loop
#todo Trainig loop je o trochu jiný pro CV a pro kasifikaci
#_ Loop trough epochs - projít epochy
#_ Loop trough trainig batches, perform traing step, calculate rain loss 
#_  per batch - projít dávky, zoptimalizovat data, spočítat loss
#_ Loop trough testing batches, perform traing step, calculate rain loss 
#_  per batch - to stejné akorát pro testové dávky
#_ Vizualizae
#_ Zmněřit čas 

#! Seed + zapnuti timeru
torch.manual_seed(42)
train_time_start_on_cpu = timer()

#! Trainig loop
epochs = 3

for epochs in tqdm(range(epochs)):
    print(f" Epoch: {epochs}\n======")
    #? Trainig
    train_loss = 0
    # přidá loop pro přidání loss pro batche
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        #? Forward pass
        y_pred = model_0(X)
        #? Loss per bach
        loss = loss_fn(y_pred,y)
        train_loss += loss      #? přidává do promněné 
        #? Optimize zero grad
        optimizer.zero_grad()
        #? Back propagation
        loss.backward()
        #? Optimizer step
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch*len(X)}/{len(train_dataloader.dataset)} sampels.")

    #? Dělení train_loss  dékou dataloaderu(batches)
    train_loss /= len(train_dataloader)
    #! Test loop
    test_loss,test_acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            #? Forward pass
            test_pred = model_0(X_test)                              #!
            #? Test loss
            test_loss += loss_fn(test_pred,y_test)                    #!
            #? Přesnost
            test_acc += accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))   #!
        #? Průměená test loss
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    #! Vizualizace
print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
#! Vypnutí timeru
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))

#! Vytváření predikcí
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """ Vrací dictonary s výsledky modelu předpovídajícího na data_loader"""
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            #? Forward pass -> predikce
            y_pred = model(X)
            #? kumulace loss a acc za každé opakování
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true= y,
                               y_pred=y_pred.argmax(dim=1))
            #? Prumněrná loss a acc
        loss /=len(data_loader)
        acc /=len(data_loader)
    return{"model_name": model.__class__.__name__,
           "model_loss": loss.item(),
           "model_acc": acc}
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn = accuracy_fn)
print(model_0_results)

#! Převod na správné zařízení ( místo kde se bude model pocitat)

device = "cuda" if torch.cuda.is_available() else "cpu"







