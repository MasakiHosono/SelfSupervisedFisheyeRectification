import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

from models import ParametersEstimationModule
from core.config import Config
from datasets import DistortDataset, BaseDataset, FisheyeEffector, EvalDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help='cfg/****.yml')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input directory which contains images.')
parser.add_argument('-ow', '--output_width', type=int, required=False, default=1280, help='Width of the output image.')
parser.add_argument('-oh', '--output_height', type=int, required=False, default=720, help='Height of the output image.')

def main(args):
    config = Config(args.config_file).getDict()

    data_path = args.input_dir
    if not os.path.exists(data_path):
        print('ERROR: No such file or directory: {}'.format(args.input_dir))
        exit(1)

    in_channels = 3

    model = ParametersEstimationModule(in_channels=in_channels).to(DEVICE)
    model = model.eval()
    transform = model.getTransforms()

    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)

    output_dir = os.path.join(data_path, 'out')
    os.makedirs(output_dir, exist_ok=True)

    dataset = EvalDataset(
        data_path = data_path,
        transform = transform
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
        for image, path in dataloader:
            output = model(image.to(DEVICE))
            output = output.data.to('cpu').item()

            print(f'Got {output} for {path}')

            filename = os.path.basename(path[0])

            effector = FisheyeEffector(
                height = args.output_height,
                width = args.output_width,
                distortion = output,
            )

            image = image[0]

            # denormalize
            image = image * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
            image = image + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

            image = transforms.ToPILImage(mode='RGB')(image)
            image = effector(image)
            image.save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    main(parser.parse_args())
