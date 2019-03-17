import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import research.common.utils as utils
import research.common.fabric_models as fm

import albumentations
from albumentations import torch as AT
from research.common.loader import CancerDataset
sys.path.append('../../common')

n_class = 2

is_available_cuda = True

# For ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b',
                        help='Batch_size',
                        type=int,
                        default=56)
    parser.add_argument('--dest', '-d',
                        help='Path to save model')
    parser.add_argument('--src', '-s',
                        help='Path to model',
                        default='/home/maxim/Work/CancerDetection/data/dense_122.pt')
    parser.add_argument('--weight', '-w',
                        help='weight image size for net',
                        type=int, default=224)
    parser.add_argument('--height',
                        help='height image size for net',
                        type=int, default=224)
    parser.add_argument('--labels', '-l',
                        help='Path to labels of classes')
    parser.add_argument('--weights',
                        help='Path to weights of classes')
    parser.add_argument('--train_csv',
                        help='Path to data CSV',
                        default='/home/maxim/Work/CancerDetection/data/my_valid.csv')
    parser.add_argument('--data_train',
                        help='Path to images for dataset',
                        default='/home/maxim/Work/CancerDetection/data/train/')
    parser.add_argument('--validate_csv',
                        help='Path to  validate data CSV ',
                        default='/home/maxim/Work/CancerDetection/data/my_valid.csv')
    parser.add_argument('--data_validate',
                        help='Path to images for dataset validate',
                        default='/home/maxim/Work/CancerDetection/data/train/')
    parser.add_argument('--cuda', dest='is_available_cuda',
                        action='store_true')
    parser.add_argument('--no-cuda', dest='is_available_cuda',
                        action='store_false')
    parser.set_defaults(is_available_cuda=True)
    args = parser.parse_args()
    best_accuracy = -1.0

    data_transforms = albumentations.Compose([
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(),
            albumentations.RandomBrightness(), albumentations.RandomContrast(),
            albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5),
        albumentations.HueSaturationValue(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    data_transforms_valid = albumentations.Compose([
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    '''data_transforms = T.Compose([
        #T.Resize((args.weight, args.height)),
        T.ColorJitter(brightness=0.5,
                      contrast=0.5),
        T.RandomRotation((-90, 90)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std)]
    )

    data_transforms_valid = T.Compose([
        #T.Resize((args.weight, args.height)),
        T.ToTensor(),
        T.Normalize(mean, std)]
    )'''

    model = fm.get_dense_net_169(n_class, pretrained=False)

    model.load_state_dict(torch.load(args.src))

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    writer = SummaryWriter()
    train_ds = CancerDataset(args.train_csv,
                             args.data_train,
                             transform_image=data_transforms,
                             )
    valid_ds = CancerDataset(args.validate_csv,
                             args.data_validate,
                             transform_image=data_transforms_valid
                             )

    loader_validate = DataLoader(valid_ds,
                                 batch_size=args.batch_size, num_workers=1)
    loader_train = DataLoader(train_ds,
                              batch_size=args.batch_size, num_workers=1)

    for epoch in range(100):

        utils.train(model, writer, is_available_cuda, loader_train,
                    F.cross_entropy, optimizer, epoch + 122)
        accuracy = utils.validate(model, writer, is_available_cuda,
                                  loader_validate, F.cross_entropy, epoch + 122)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
            best_model.cpu()
            pth_model = '/home/maxim/Work/CancerDetection/data/dense_{0}.pt'.format(epoch + 122)
            torch.save(best_model.state_dict(), pth_model)


if __name__ == '__main__':
    main()
