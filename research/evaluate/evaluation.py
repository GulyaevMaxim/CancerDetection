import torch
import torchvision.transforms as T
import argparse
from tqdm import tqdm
import sys
import pandas
from torch.utils.data import DataLoader
import research.common.utils as utils
from research.common.loader import CancerDataset
import numpy
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('../../common')

is_available_cuda = True

# For ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#from pytorchcv.model_provider import get_model as ptcv_get_model
#net = ptcv_get_model("mobilenet_w1", pretrained=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b',
                        help='Batch_size',
                        type=int,
                        default=80)
    parser.add_argument('--weight', '-w',
                        help='weight image size for net',
                        type=int, default=224)
    parser.add_argument('--height',
                        help='height image size for net',
                        type=int, default=224)
    parser.add_argument('--model', '-m',
                        help='Path to model')
    parser.add_argument('--data_csv',
                        help='Path to data CSV')
    parser.add_argument('--data',
                        help='Path to images for dataset')
    parser.add_argument('--path_to_out_zip',
                        help='Path to  out file with submission ')
    parser.add_argument('--cuda', dest='is_available_cuda',
                        action='store_true')
    parser.add_argument('--no-cuda', dest='is_available_cuda',
                        action='store_false')
    parser.set_defaults(is_available_cuda=True)
    args = parser.parse_args()
    path_to_out = args.path_to_out_zip

    image_size = (args.weight, args.height)

    test_ds = CancerDataset(csv_file=args.data_csv,
                            root_dir=args.data,
                            transform_image=T.Compose([
                                T.Resize(image_size),
                                T.ToTensor(),
                                T.Normalize(mean, std)
                            ]))

    loader_test = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=1)

    submission_names = test_ds.get_train_img_names()
    model = torchvision.models.densenet169(pretrained='imagenet')
    # for child in model.children():
    #    for param in child.parameters():
    #        param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if is_available_cuda:
        model.cuda()

    predicted_labels = []
    pbar = tqdm(loader_test)
    for batch_idx, data in enumerate(pbar):
        with torch.no_grad():

            if is_available_cuda:
                data = Variable(data[0].cuda(), requires_grad=False)
            else:
                data = Variable(data[0], requires_grad=False)

            y_predicted = model(data)
            for predicted in y_predicted:

                label = numpy.argmax(predicted.cpu().numpy())
                predicted_labels.append(label)
            del data
            del y_predicted

    df = pandas.DataFrame({'id': submission_names, 'label': predicted_labels})
    df.to_csv('{}.gz'.format(path_to_out), index=False,
              compression='gzip')


if __name__ == '__main__':
    main()
