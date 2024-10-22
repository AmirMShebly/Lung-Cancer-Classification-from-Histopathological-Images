import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LungCancerDataset(Dataset):
    def __init__(self, root, transform=None, train=True, split_ratio=0.8):
        self.root = root
        self.transform = transform
        self.classes = ['adenocarcinoma', 'benign', 'squamous_cell_carcinoma']

        self.image_paths = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root, class_name)
            images = os.listdir(class_dir)
            image_paths = [os.path.join(class_dir, img) for img in images]
            
            random.shuffle(image_paths)
            split_idx = int(split_ratio * len(image_paths))

            if train:
                self.image_paths += image_paths[:split_idx]
            else:
                self.image_paths += image_paths[split_idx:]

            self.labels += [class_idx] * (split_idx if train else len(image_paths) - split_idx)

    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        imag_path = self.image_paths[idx]
        image = Image.open(imag_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
