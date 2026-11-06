
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
class VinDrDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        self.hard_negatives = set()
        if self.is_train and os.path.exists("hard_negatives_to_mine.txt"):
            with open("hard_negatives_to_mine.txt", "r") as f:
                self.hard_negatives = set(line.strip() for line in f)
            
        self.img_col = next((col for col in self.df.columns if 'path' in col.lower() or 'image' in col.lower()), self.df.columns[0])  # [clipped] ending restored
        
        exclude_cols = {self.img_col, 'Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position'}  # [clipped] tail restored, check your column names
        self.label_cols = [col for col in self.df.columns if col not in exclude_cols]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        
        
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
          
            for candidate in [img_name, os.path.join("images", img_name), os.path.join("../images", img_name), img_name]:  
                    img_path = candidate
                    break
            
        image = Image.open(img_path).convert("RGB")
        
        labels = row[self.label_cols].values.astype(np.float32)
        
        if self.is_train and img_name in self.hard_negatives:
            labels = np.zeros_like(labels)
            
        if self.transform:
            image = self.transform(image)
