#!/usr/bin/env python


from __future__ import print_function

import os
import pandas
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class CancerDataset(Dataset):

    def __init__(self, csv_file,
                 root_dir,
                 transform_image=None,
                 ):
        """

        Parameters
        ----------
        csv_file Path to the csv file with annotations.
        root_dir Directory with all the images.
        mask_dir Directory with all the masks.
        transform_image transforms for images
        transform_mask transforms for masks
        """
        self.train_images = pandas.read_csv(csv_file)

        self.train_img_names = self.train_images.values[:, 1]
        self.predicted_labels = self.train_images.values[:, 2]

        self.root_dir = os.path.abspath(root_dir)

        self.transform_image = transform_image

    def __len__(self):
        return len(self.train_img_names)

    def __getitem__(self, id):
        img_name = os.path.join(self.root_dir, self.train_img_names[id])
        img_name += '.tif'
        image = Image.open(img_name).convert('RGB')

        if self.transform_image:
            image = self.transform_image(image)

        label_id = self.predicted_labels[id]

        return image, label_id

    def get_train_images(self):
        return self.train_images

    def get_train_img_names(self):
        return self.train_img_names





