import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ParametersEstimationModule(nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, 1),
                nn.Conv2d(64, 64, 3, 1),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1),
                nn.Conv2d(128, 128, 3, 1),
                nn.MaxPool2d(2, 2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, 2),
                nn.ConvTranspose2d(64, 3, 2, 2),
            )
            self.vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=False)
            self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.vgg(x)
            x = torch.flatten(x)
            return x

        def getTransforms(self):
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
