import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
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
sys.path.append('../../common')
from research.common.loader import CancerDataset

#from pytorchcv.model_provider import get_model as ptcv_get_model
#net = ptcv_get_model("fishnet150", pretrained=True)


n_class = 2

is_available_cuda = True

# For ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

pth_model = '/home/maxim/Work/CancerDetection/data/r18.pt'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b',
                        help='Batch_size',
                        type=int,
                        default=80)
    parser.add_argument('--dest', '-d',
                        help='Path to save model')
    parser.add_argument('--src', '-s',
                        help='Path to model',
                        default=None)
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
                        help='Path to data CSV')
    parser.add_argument('--data_train',
                        help='Path to images for dataset')
    parser.add_argument('--validate_csv',
                        help='Path to  validate data CSV ')
    parser.add_argument('--data_validate',
                        help='Path to images for dataset validate')
    parser.add_argument('--cuda', dest='is_available_cuda',
                        action='store_true')
    parser.add_argument('--no-cuda', dest='is_available_cuda',
                        action='store_false')
    parser.set_defaults(is_available_cuda=True)
    args = parser.parse_args()
    best_accuracy = -1.0

    data_transforms = T.Compose([
        T.Resize((args.weight, args.height)),
        T.ColorJitter(brightness=0.5,
                      contrast=0.5),
        T.RandomRotation((0, 5)),
        T.ToTensor(),
        T.Normalize(mean, std)])

    data_transforms_valid = T.Compose([
        T.Resize((args.weight, args.height)),
        T.ToTensor(),
        T.Normalize(mean, std)])

    model = torchvision.models.resnet18(pretrained='imagenet')
    #for child in model.children():
    #    for param in child.parameters():
    #        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)

    #model.load_state_dict(torch.load(pth_model))

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    writer = SummaryWriter()
    train_ds = CancerDataset('/home/maxim/Work/CancerDetection/data/my_train.csv',
                             '/home/maxim/Work/CancerDetection/data/train/',
                             transform_image=data_transforms,
                             )
    valid_ds = CancerDataset('/home/maxim/Work/CancerDetection/data/my_valid.csv',
                             '/home/maxim/Work/CancerDetection/data/train/',
                             transform_image=data_transforms_valid
                             )

    loader_validate = DataLoader(valid_ds,
                                 batch_size=args.batch_size, num_workers=1)
    loader_train = DataLoader(train_ds,
                              batch_size=args.batch_size, num_workers=1)

    for epoch in range(1000):

        utils.train(model, writer, is_available_cuda, loader_train,
                    F.cross_entropy, optimizer, epoch)
        accuracy = utils.validate(model, writer, is_available_cuda,
                                  loader_validate, F.cross_entropy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
            best_model.cpu()
            torch.save(best_model.state_dict(), pth_model)


if __name__ == '__main__':
    main()
