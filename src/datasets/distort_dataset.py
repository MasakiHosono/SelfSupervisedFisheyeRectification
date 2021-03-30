import os
import io
import cv2
import random
import torch
from PIL import Image
from . import BaseDataset
from . import FisheyeEffector

class DistortDataset(BaseDataset):
    '''
    Expected Dataset Format is:
    DATASET_ROOT/
        train.lst
        val.lst
        test.lst
        hoge.jpg
        fuga.jpg
        piyo.jpg
        ...
    '''
    def __init__(self, list_path, height=720, width=1280, transform=None, distortions=[], return_distortion=False, output_dir=None):
        super().__init__(transform=transform)
        self.return_distortion = return_distortion
        self.height, self.width = height, width
        self.output_dir = output_dir

        if len(distortions) > 0:
            self.effectors = [
                FisheyeEffector(height=height, width=width, distortion=d)
                for d in distortions
            ]
        else:
            self.effectors = [
                FisheyeEffector(height=height, width=width, distortion=d)
                for d in [random.uniform(-1, 1) for _ in range(10)]
            ]

        self.list_path = list_path
        self.img_list = []
        if self.list_path is not None:
            self.img_list = [line.strip() for line in open(list_path)]

    def updateEffector(self, distortions=[]):
        if len(distortions) > 0:
            self.effectors = [
                FisheyeEffector(height=self.height, width=self.width, distortion=d)
                for d in distortions
            ]
        else:
            self.effectors = [
                FisheyeEffector(height=self.height, width=self.width, distortion=d)
                for d in [random.uniform(-1, 1) for _ in range(10)]
            ]

    def __getitem__(self, idx):
        image_file = self.img_list[idx]
        effector = random.choice(self.effectors)

        image = Image.open(image_file)
        image = image.resize((self.width, self.height))
        if self.output_dir:
            image.save(os.path.join(self.output_dir, f'{idx}_org.jpg'))

        image = effector(image)
        if self.output_dir:
            image.save(os.path.join(self.output_dir, f'{idx}_dis.jpg'))

        if self.transform:
            image = self.transform(image)

        if self.return_distortion:
            return image, effector.getDistortion()

        return image, effector.getKeyCoordinates()
