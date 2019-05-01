import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import albumentations
from albumentations import torch as AT

path = os.path.abspath(__file__).rsplit(os.path.sep, 2)[0]
path = os.path.join(path, 'common')
sys.path.append(path)
import utils
import fabric_models as fm
from loader import CancerDataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        help='Settings file')
    args = parser.parse_args()
    with open(args.s) as config_file:
        config = json.load(config_file)

    best_accuracy = -1.0
    save_path = config.get('dest_model')

    model = fm.get_dense_net_121(pretrained=True)

    src_path = config.get('src_model')
    if src_path is not None:
        model.load_state_dict(torch.load(src_path))

    is_available_cuda = config.get('cuda', False)
    if is_available_cuda:
        model.cuda()

    data_transforms = albumentations.Compose([
        albumentations.Resize(int(config.get('width')),
                              int(config.get('height'))),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2),
            albumentations.IAASharpen(),
            albumentations.IAAEmboss(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            albumentations.JpegCompression(),
            albumentations.Blur(),
            albumentations.GaussNoise()], p=0.5),
        albumentations.HueSaturationValue(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.15,
                                        scale_limit=0.15,
                                        rotate_limit=45, p=0.5),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    data_transforms_valid = albumentations.Compose([
        albumentations.Resize(int(config.get('width')),
                              int(config.get('height'))),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    writer = SummaryWriter()
    train_ds = CancerDataset(csv_file=config.get('train_csv'),
                             root_dir=config.get('data_train'),
                             transform_image=data_transforms,
                             )
    valid_ds = CancerDataset(csv_file=config.get('valid_csv'),
                             root_dir=config.get('data_validate'),
                             transform_image=data_transforms_valid
                             )

    loader_validate = DataLoader(valid_ds,
                                 batch_size=config.get('batch_size'),
                                 num_workers=1)
    loader_train = DataLoader(train_ds,
                              batch_size=config.get('batch_size'),
                              num_workers=1)

    criterion = nn.BCEWithLogitsLoss()
    start_epoch = config.get('st_epoch', 0)
    for epoch in range(config.get('do_epoch', 1000)):

        utils.train(model, writer, is_available_cuda, loader_train,
                    criterion, optimizer, epoch + start_epoch)
        accuracy = utils.validate(model, writer, is_available_cuda,
                                  loader_validate, criterion, epoch + start_epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
            best_model.cpu()
            pth_model = '{0}_{1}.pt'.format(save_path, epoch + start_epoch)
            torch.save(best_model.state_dict(), pth_model)


if __name__ == '__main__':
    main()