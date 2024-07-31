import torchvision
import torch
from torch import nn
from torchvision import  transforms
import matplotlib.pyplot as plt
from typing import List,Tuple
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def zobraz_fotku(image_path:str,
                 model:nn.Module,
                 model_save_path:str,
                 class_names:list,
                 device:torch.device = device):
    model.load_state_dict(torch.load(f=model_save_path))
    image = torchvision.io.read_image(path=str(image_path)).type(torch.float32)/255
    transformed_image = transforms.Compose([transforms.Resize(size=(64,64))])
    transformed_image = transformed_image(image)
    model.eval()
    with torch.inference_mode():
        image_pred = model(transformed_image.unsqueeze(dim=0).to(device))
    image_pred_probs = torch.softmax(image_pred,dim=1)
    image_pred_labels = torch.argmax(image_pred_probs)

    image_vizualization = torchvision.io.read_image(str(image_path))
    plt.imshow(image_vizualization.permute(1,2,0))
    plt.title(f"Jmeno jidla je: {class_names[image_pred_labels]}")
    plt.show()

def pred_zobraz_fotku_PIL(model:nn.Module,
                     class_names:List[str],
                     image_path:str,
                     image_size: Tuple[int, int] = (224, 224),
                     transform: torchvision.transforms = None,
                     device: torch.device=device):
    
    img = Image.open(image_path)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([transforms.Resize(image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
        ])
    model.to(device)
    model.eval()
    with torch.inference_mode():
      #? přidá dimenzi do listu pro batch size
      transformed_image = image_transform(img).unsqueeze(dim=0)
      target_image_pred = model(transformed_image.to(device))
    #? udělá z predikcí pravděpodobnost
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    #? určí hodnotu s nejvyší pravěpodobností jako výsledek
    target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_labels]} | Prob: {target_image_pred_probs.max():.3f}")

    plt.show()