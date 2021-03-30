import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms

from models import ParametersEstimationModule
from core.functions import train, val, getDistortions
from core.losses import DistortionLoss
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
    transform = model.getTransforms()

    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)

    summary(model, input_size=(in_channels, config['DATASET']['HEIGHT'], config['DATASET']['WIDTH']))

    criterion = DistortionLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['TRAIN']['LEARNING_RATE'])

    max_epoch = config['TRAIN']['MAX_EPOCH']
    last_epoch = 0
    train_losses = []
    val_losses = []
    enable_curriculum = config['TRAIN']['CURRICULUM']['ENABLED']
    switch_epoch = config['TRAIN']['CURRICULUM']['SWITCH_EPOCH']

    output_dir = os.path.join('outputs', config['DATASET']['NAME'])
    model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(model_state_file):
        checkpoint = torch.load(model_state_file)
        last_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint (epoch {})'.format(last_epoch))

    if enable_curriculum:
        num_patterns = int(last_epoch / switch_epoch) + 2
        if num_patterns <= 10:
            distortions = getDistortions(num_patterns, random_values=False)
        else:
            distortions = getDistortions(10, random_values=True)
    else:
        distortions = getDistortions(10, random_values=True)

    trainset = DistortDataset(
        list_path = os.path.join(data_path, 'train.lst'),
        height = config['DATASET']['HEIGHT'],
        width = config['DATASET']['WIDTH'],
        transform = transform,
        distortions = distortions,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = config['TRAIN']['BATCH_SIZE'],
        shuffle = True,
        num_workers = 1
    )

    testset = DistortDataset(
        list_path = os.path.join(data_path, 'val.lst'),
        height = config['DATASET']['HEIGHT'],
        width = config['DATASET']['WIDTH'],
        transform = transform,
        distortions = distortions,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = config['TRAIN']['BATCH_SIZE'],
        shuffle = True,
        num_workers = 1
    )

    for epoch in range(last_epoch, max_epoch):
        print('Epoch {}'.format(epoch+1))

        train_loss = train(
            model = model,
            dataloader = trainloader,
            criterion = criterion,
            optimizer = optimizer,
            device = DEVICE,
        )
        val_loss = val(
            model = model,
            dataloader = testloader,
            criterion = criterion,
            device = DEVICE,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print('Loss: train = {}, val = {}'.format(train_loss, val_loss))

        plt.plot(range(epoch+1), train_losses, label="train")
        plt.plot(range(epoch+1), val_losses, label="val")
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'losses.png'))
        plt.clf()

        if (epoch+1) % switch_epoch == 0:
            num_patterns = int((epoch+1) / switch_epoch) + 2
            if enable_curriculum and num_patterns <= 10:
                distortions = getDistortions(num_patterns, random_values=False)
            else:
                distortions = getDistortions(10, random_values=True)
            trainset.updateEffector(distortions=distortions)
            testset.updateEffector(distortions=distortions)

        print('=> saving checkpoint to {}'.format(model_state_file))
        torch.save(
            {
                'epoch': epoch+1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            model_state_file
        )

if __name__ == '__main__':
    main(parser.parse_args())
