import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data

import numpy as np
# from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class Potsdam(data.Dataset):
    """Potsdam ISPRS Dataset <http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html>.
    
    Parameters:
        - root: Root directory containing imgs and labels directories.
        - transform: A function that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    PotsdamClass = namedtuple('PotsdamClass', ['name', 'id', 'ignore_in_eval', 'color'])
    classes = [
        PotsdamClass('background',    0, True, (255, 0, 0)),      # red
        PotsdamClass('building',      1, False, (0, 0, 255)),     # blue
        PotsdamClass('pavement',      2, False, (255, 255, 255)), # white
        PotsdamClass('vegetation',    3, False, (0, 255, 255)),   # turqoise
        PotsdamClass('tree',          4, False, (0, 255, 0)),     # green
        PotsdamClass('car',           5, False, (255, 255, 0)),   # yellow
    ]

    id_to_color = np.array([c.color for c in classes])
    color_to_id = {c.color:c.id for c in classes}
    
    def __init__(self, root, list_IDs, transform=None):
        self.root = os.path.expanduser(root)
        self.imgs_dir = os.path.join(self.root, 'imgs')
        self.targets_dir = os.path.join(self.root, 'masks')

        self.list_IDs = list_IDs # same ID for input and label
        self.transform = transform
        
        self.images = []
        self.targets = []
        for ID in self.list_IDs:
            X = np.load(os.path.join(self.imgs_dir, ID + '.npy'))
            y = np.load(os.path.join(self.targets_dir, ID + '.npy'))

            self.images.append(X)
            self.targets.append(y)
        
    @classmethod
    def encode_target(cls, target):
        # expects target mask to be (H, W, C)
        enc = np.zeros(target.shape[:2], dtype=np.int8)
        for i, row in enumerate(target):
            for j, color in enumerate(row):
                try: # look up id
                    ID = cls.color_to_id[tuple(color)]
                except: # catch occasional invalid color value
                    inval_idx = [np.all(a) for a in zip(color!=0, color!=255)]
                    color[inval_idx] = 255
                    ID = cls.color_to_id[tuple(color)]
                enc[i,j] = ID
        return enc.astype(np.int32)
#         return np.array([[cls.color_to_id[tuple(color)] for color in row] for row in target])

    @classmethod
    def decode_target(cls, target):
        dec = cls.id_to_color[target]
        return dec

    def __getitem__(self, index):
        X, y = self.images[index], self.targets[index]
        
        # Augment data
        if self.transform:
            segmap = SegmentationMapsOnImage(y, shape=y.shape)
            X, segmap = self.transform(image=X, segmentation_maps=segmap)
            y = segmap.get_arr()
            
        # Normalize image with ImageNet mean / std
        X = X / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        X = (X - mean) / std
        
        # Expect some pixels now negative
#         print("Mean and standard deviation are:", X.mean(), X.std())
        
        # Switch HWC images to CHW pytorch order
        X = X.transpose((2, 0, 1))
        
        return X, y
       
    def __len__(self):
        return len(self.list_IDs)