import os
import torch
from PIL import Image
from . import BaseDataset

class EvalDataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform=transform)

        for file in os.listdir(data_path):
            if os.path.isfile(os.path.join(data_path, file)):
                self.img_list.append(os.path.join(data_path, file))

    def __getitem__(self, idx):
        image_file = self.img_list[idx]

        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)

        return image, image_file
