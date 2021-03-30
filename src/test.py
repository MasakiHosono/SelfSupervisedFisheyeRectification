import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

from models import ParametersEstimationModule
from core.functions import getDistortions
from core.config import Config
from datasets import DistortDataset, FisheyeEffector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help='cfg/****.yml')

def main(args):
    config = Config(args.config_file).getDict()

    data_path = os.path.join('data', config['DATASET']['NAME'])
    if not os.path.exists(data_path):
        print('ERROR: No dataset named {}'.format(config['DATASET']['NAME']))
        exit(1)

    in_channels = 3

    model = ParametersEstimationModule(in_channels=in_channels).to(DEVICE)
    model = model.eval()
    transform = model.getTransforms()

    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)

    distortions = getDistortions(11)

    output_dir = None
    if config['TEST']['SAVE_RESULTS']:
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)

    dataset = DistortDataset(
        list_path = os.path.join(data_path, 'test.lst'),
        height = config['DATASET']['HEIGHT'],
        width = config['DATASET']['WIDTH'],
        transform = transform,
        distortions = distortions,
        return_distortion = True,
        output_dir = output_dir
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )

    model_file = os.path.join('outputs', config['DATASET']['NAME'], config['TEST']['CHECKPOINT'])

    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print('=> load {}'.format(model_file))

    else:
        print('model_file "{}" does not exists.'.format(model_file))
        exit(1)

    with torch.no_grad():
        for idx, (data, distortion) in enumerate(dataloader):
            outputs = model(data.to(DEVICE))
            outputs = outputs.data.to('cpu').item()

            print(f'Got {outputs}, distorted by {distortion.item()}')

            if output_dir is not None:
                effector = FisheyeEffector(
                    height = config['TEST']['OUTPUT_SIZE']['HEIGHT'],
                    width = config['TEST']['OUTPUT_SIZE']['WIDTH'],
                    distortion = outputs,
                )

                # denormalize
                image = data[0]
                image = image * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
                image = image + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

                image = transforms.ToPILImage(mode='RGB')(image)
                image = effector(image)
                image.save(os.path.join(output_dir, f'{idx}_rec.jpg'))

if __name__ == '__main__':
    main(parser.parse_args())
