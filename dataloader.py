import os 
import numpy as np 
import torch 
from torch.utils.data import Dataset,transforms
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from PIL import Image 
#dataset training 
img_size = (128 , 128)
class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir,tranform_img= None,transform_mask=None):
        self.input_dir  = input_dir
        self.target_dir = target_dir
        self.input_img_paths =sorted([os.path.join(input_dir , fname )for fname in os.listdir(self.input_dir) if fname.endswith(".jpg")])
        self.target_img_paths = sorted([os.path.join(target_dir , fname) for fname in os.listdir(self.target_dir) if fname.endswith(".png") and not fname.startswith(".")])
        self.tranforms_img - tranform_img 
        self.transform_mask = transform_mask
        self.data = data
        

    def __len__(self):
        return len(self.input_img_paths)
    
    def __get__item(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        mask_path = self.target_img_paths[idx]
        mask = Image.open(mask_path).convert("L")
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        return img, mask
        
transform_img = transforms.Compose([
    transforms.Resize(img_size),    # Resize images to the desired size
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

transform_mask = transforms.Compose([
    transforms.Resize(img_size),    # Resize masks to the desired size
    transforms.ToTensor()           # Convert the mask to a tensor
])

# Load the dataset
def create_splits(dataset, num_splits=5):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, shuffle=True)
    splits = torch.chunk(torch.tensor(train_indices), num_splits)
    return splits, val_indices

def load_custom_dataset(split_id, splits, val_indices, dataset):
    # Load the specific train split for this client
    train_subset = Subset(dataset, splits[split_id].tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    return trainloader, valloader
    
