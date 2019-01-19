import os

import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class DatasetFetcher:
    def __init__(self):

        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None
        self.valid_data = None
        self.valid_masks_data = None
        self.valid_files = None
        self.valid_masks_files = None

    def fetch_dataset(self):

        project_root = os.path.dirname(os.path.abspath(__file__))
        destination_path = os.path.join(project_root + "/../../input")

        datasets_path = [destination_path + "/train", destination_path + "/test",
                         destination_path + "/train_masks", destination_path + "/valid",
                         destination_path + "/valid_masks"]

        self.train_data = datasets_path[0]
        self.test_data = datasets_path[1]
        self.train_masks_data = datasets_path[2]
        self.valid_data = datasets_path[3]
        self.valid_masks_data = datasets_path[4]
        self.train_files = sorted(os.listdir(self.train_data))
        self.valid_files = sorted(os.listdir(self.valid_data))
        self.test_files = sorted(os.listdir(self.test_data))
        self.train_masks_files = sorted(os.listdir(self.train_masks_data))
        self.valid_masks_files = sorted(os.listdir(self.valid_masks_data))
        return datasets_path

    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.float32)

    def get_mask_matrix(self, mask_path):
        mask = Image.open(mask_path)
        return np.asarray(mask, dtype=np.int64)

    def get_train_files(self):

        train_files = self.train_files
        # train_files = [os.path.splitext(file)[0] for file in self.train_files];

        train_ret = [self.train_data + "/" + s for s in train_files]
        train_masks_ret = [self.train_masks_data + "/" + s for s in train_files]

        return [np.array(train_ret).ravel(), np.array(train_masks_ret).ravel()]

    def get_valid_files(self):

        valid_files = self.valid_files
        # valid_files = [os.path.splitext(file)[0] for file in self.valid_files];

        valid_ret = [self.valid_data + "/" + s for s in valid_files]
        valid_masks_ret = [self.valid_masks_data + "/" + s for s in valid_files]

        return [np.array(valid_ret).ravel(), np.array(valid_masks_ret).ravel()]

    def get_test_files(self):
        test_files = self.test_files

        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_data + "/" + file

        return np.array(ret)
