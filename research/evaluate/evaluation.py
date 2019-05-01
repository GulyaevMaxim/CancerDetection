import torch
import torchvision
import argparse
import sys
import os
import pandas
import numpy
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import albumentations
from albumentations import torch as AT

path = os.path.abspath(__file__).rsplit(os.path.sep, 2)[0]
path = os.path.join(path, 'common')
sys.path.append(path)
import utils
from loader import CancerDataset
import fabric_models as fm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        help='Settings file')
    args = parser.parse_args()
    with open(args.s) as config_file:
        config = json.load(config_file)

    path_to_out = config.get('out_path')
    is_available_cuda = config.get('cuda', False)

    test_ds = CancerDataset(csv_file=config.get('data_csv'),
                            root_dir=config.get('data'),
                            transform_image=albumentations.Compose([
                                albumentations.Resize(int(config.get('width')),
                                                      int(config.get('height'))),
                                albumentations.Normalize(),
                                AT.ToTensor()
                            ]))

    loader_test = DataLoader(test_ds, batch_size=config.get('batch_size'),
                             num_workers=1)

    submission_names = test_ds.get_train_img_names()
    model = fm.get_dense_net_121(pretrained=False)

    model.load_state_dict(torch.load(config.get('model')))
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
            y_predicted = torch.sigmoid(y_predicted)

            for predicted in y_predicted:

                predicted_labels.append(predicted.cpu().numpy()[0])

            del data
            del y_predicted

    predicted_labels = numpy.array(predicted_labels)
    utils.generate_submission(submission_names,
                              predicted_labels, path_to_out)


if __name__ == '__main__':
    main()
