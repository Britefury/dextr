# The MIT License (MIT)
#
# Copyright (c) 2020 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.

"""
Pascal VOC dataset
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

    def __init__(self, split, transform, load_input=True, progress_fn=None):
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

        super(PascalVOCDataset, self).__init__(obj_meta_path, transform, load_input=load_input,
                                               progress_fn=progress_fn)


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
