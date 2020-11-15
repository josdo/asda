import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
import torchvision

import numpy as np
from PIL import Image
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
    
    def __init__(self, root, list_path, transform=None):
        self.root = os.path.expanduser(root)
        self.imgs_dir = os.path.join(self.root, 'imgs')
        self.targets_dir = os.path.join(self.root, 'masks')

        self.list_IDs = [i_id.strip() for i_id in open(list_path)] # same ID for input and label
        self.transform = transform

        # ImageNet mean and std
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        
        self.files = []
        for ID in self.list_IDs:
            # X = np.array(Image.open(os.path.join(self.imgs_dir, ID)).convert('RGB'))
            # y = np.array(Image.open(os.path.join(self.targets_dir, ID)).convert('RGB'))

            self.files.append({
                "img": os.path.join(self.imgs_dir, ID + '.png'), # X,
                "label": os.path.join(self.targets_dir, ID + '.npy'), # y
                "name": ID,
            })
        
    @classmethod
    def encode_target(cls, target):
        # expects target mask to be (H, W, C)
        target = (np.around(target / 255) * 255).astype(np.uint8)  # round colors
        enc = np.zeros(target.shape[:2])
        for potsdam_class in cls.classes:
            label = potsdam_class.id
            color = np.asarray(potsdam_class.color)
            enc += (target == color).all(axis=2) * label
        #for i, row in enumerate(target):
        #    for j, color in enumerate(row):
        #        try: # look up id
        #            ID = cls.color_to_id[tuple(color)]
        #        except: # catch occasional invalid color value
        #            inval_idx = [np.all(a) for a in zip(color!=0, color!=255)]
        #            color[inval_idx] = 255
        #            ID = cls.color_to_id[tuple(color)]
        #        enc[i,j] = ID
        return enc.astype(np.int32)
#         return np.array([[cls.color_to_id[tuple(color)] for color in row] for row in target])

    @classmethod
    def decode_target(cls, target):
        dec = cls.id_to_color[target]
        return dec

    def __getitem__(self, index):
        datafiles = self.files[index]
        X = np.array(Image.open(datafiles["img"]).convert('RGB'), dtype=np.float32)
        y = np.load(datafiles["label"])
        # y = np.array(Image.open(datafiles["label"]).convert('RGB'), dtype=np.uint8)
        name = datafiles["name"]
        #X, y = self.images[index], self.targets[index]
        
        # Augment data
        #if self.transform:
        #    segmap = SegmentationMapsOnImage(y, shape=y.shape)
        #    X, segmap = self.transform(image=X, segmentation_maps=segmap)
        #    y = segmap.get_arr()
        
        # Normalize image with ImageNet mean / std
        X = X / 255
        X = (X - self.mean) / self.std
        
        # Expect some pixels now negative
#         print("Mean and standard deviation are:", X.mean(), X.std())
        
        size = X.shape
        X = X[:, :, ::-1] - np.zeros_like(X)  # change to BGR
        X = X.transpose((2, 0, 1))  # Switch HWC images to CHW pytorch order
        
        return X, y, np.array(size), name
       
    def __len__(self):
        return len(self.list_IDs)

if __name__ == '__main__':
    dst = Potsdam('./data/potsdam', './dataset/potsdam_list/train.txt')
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
            img.save('Potsdam_Demo.jpg')
        break
