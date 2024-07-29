import torchvision
import torch
from torch import nn
from torchvision import  transforms
import matplotlib.pyplot as plt
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
