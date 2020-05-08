"""
Pascal VOC dataset

Place VOC2012 dataset in 'VOC2012' directory.
For training, you will need the augmented labels. Download http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip.
The folder structure should be like:
VOC2012/JPEGImages
       /SegmentationClassAug
"""
import os
import numpy as np
import settings
from PIL import Image
from dextr.data_pipeline import dextr_dataset


def _load_names(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line != '']


def _get_pascal_path(exists=False):
    return settings.get_data_path(
        config_name='pascal_voc',
        exists=exists
    )

class PascalVOCDataset (dextr_dataset.DextrDataset):
    IGNORE_INDEX = 255

    def __init__(self, split, transform, load_input=True):
        pascal_path = _get_pascal_path(exists=True)

        if split == 'train':
            names_path = os.path.join(pascal_path, 'ImageSets', 'Segmentation', 'train.txt')
        elif split == 'val':
            names_path = os.path.join(pascal_path, 'ImageSets', 'Segmentation', 'val.txt')
        else:
            raise ValueError('split should be either train or val, not {}'.format(split))

        self.sample_names = _load_names(names_path)
        self.sample_names.sort()

        self.x_paths = [os.path.join(pascal_path, 'JPEGImages', '{}.jpg'.format(name))
                        for name in self.sample_names]
        self.instance_y_paths = [os.path.join(pascal_path, 'SegmentationObject', '{}.png'.format(name))
                                 for name in self.sample_names]

        self.test_image_ndx = None

        obj_meta_path = os.path.join(pascal_path, 'dextr_objects_{}.pkl'.format(split))

        super(PascalVOCDataset, self).__init__(obj_meta_path, transform, load_input=load_input)


    @property
    def num_images(self):
        return len(self.x_paths)

    def get_input_image_pil(self, img_i):
        path = self.x_paths[img_i]
        img = Image.open(path)
        img.load()
        return img

    def get_instance_y_arr(self, sample_i):
        path = self.instance_y_paths[sample_i]
        img = Image.open(path)
        img.load()
        return np.array(img)
