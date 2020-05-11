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
import os, pickle
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import label
from PIL import Image
import torch.utils.data


class DextrDataset (torch.utils.data.Dataset):
    IGNORE_INDEX = None

    def __init__(self, object_meta_path, transform, load_input=True, progress_fn=None):
        if object_meta_path is not None and os.path.exists(object_meta_path):
            obj_meta = pickle.load(open(object_meta_path, 'rb'))
            self.obj_meta_indices = obj_meta['indices']
            self.obj_meta_outlines = obj_meta['outlines']
        else:
            self.obj_meta_indices, self.obj_meta_outlines = self._build_object_metadata(progress_fn)
            if object_meta_path is not None:
                obj_meta = dict(indices=self.obj_meta_indices, outlines=self.obj_meta_outlines)
                pickle.dump(obj_meta, open(object_meta_path, 'wb'))

        self.transform = transform

        self.load_input = load_input


    def __len__(self):
        return len(self.obj_meta_indices)

    def __getitem__(self, item):
        # Get object metadata
        img_i, label_i, region_i = self.obj_meta_indices[item]
        outline = self.obj_meta_outlines[item]

        # Get the label mask and select the region
        label_mask = self.get_label_mask(img_i, label_i)
        if region_i != -1:
            regions, num_regions = label(label_mask)
            object_mask = regions == (region_i + 1)
        else:
            object_mask = label_mask
        object_mask_u8 = object_mask.astype(np.uint8) * 255
        object_mask_pil = Image.fromarray(object_mask_u8)

        sample = dict(target_mask=object_mask_pil, target_mask_outline=outline)

        if self.load_input:
            # Get input image
            input_pil = self.get_input_image_pil(img_i)
            sample['input'] = input_pil

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def _image_indices_to_object_indices(self, image_ndx):
        if image_ndx is not None:
            image_mask = np.zeros((self.num_images,), dtype=bool)
            image_mask[image_ndx] = True
            obj_mask = image_mask[self.obj_meta_indices[:, 0]]
            return np.where(obj_mask)[0]
        else:
            return None

    def _build_object_metadata(self, progress_fn=None):
        obj_meta_indices = []
        obj_meta_outlines = []

        if progress_fn is None:
            progress_fn = lambda x: x

        for img_i in progress_fn(range(self.num_images)):
            inst_y = self.get_instance_y_arr(img_i)
            num_labels = self.get_num_labels_in_inst_y(inst_y)

            for label_i in range(num_labels):
                label_mask = self.get_label_mask_from_inst_y(inst_y, label_i)
                regions, num_regions = label(label_mask)

                for region_i in range(num_regions):
                    region_mask = regions == (region_i + 1)
                    outline = self.mask_outline(region_mask)

                    if outline is not None:
                        if num_regions > 1:
                            obj_meta_indices.append([img_i, label_i, region_i])
                        else:
                            # Only 1 region; use -1 for region index
                            obj_meta_indices.append([img_i, label_i, -1])
                        obj_meta_outlines.append(outline)

        obj_meta_indices = np.array(obj_meta_indices)

        return obj_meta_indices, obj_meta_outlines

    @property
    def num_images(self):
        raise NotImplementedError

    def get_input_image_pil(self, img_i):
        raise NotImplementedError

    def get_instance_y_arr(self, img_i):
        raise NotImplementedError

    @staticmethod
    def get_label_mask_from_inst_y(inst_y, object_i):
        if inst_y.ndim == 2:
            return inst_y == (object_i + 1)
        elif inst_y.ndim == 3:
            return inst_y[:, :, object_i]
        else:
            raise ValueError('inst_y should be a 2D `(H,W)` label image or a 3D `(H,W,OBJ)` mask stack')

    @classmethod
    def get_num_labels_in_inst_y(cls, inst_y):
        if inst_y.ndim == 2:
            if cls.IGNORE_INDEX is not None:
                inst_y = inst_y[inst_y != cls.IGNORE_INDEX]
            return inst_y.max()
        elif inst_y.ndim == 3:
            return inst_y.shape[2]
        else:
            raise ValueError('inst_y should be a 2D `(H,W)` label image or a 3D `(H,W,OBJ)` mask stack')

    def get_label_mask(self, img_i, label_i):
        inst_y = self.get_instance_y_arr(img_i)
        return self.get_label_mask_from_inst_y(inst_y, label_i)

    @staticmethod
    def mask_outline(mask):
        outline = mask & ~binary_erosion(mask)
        m_y, m_x = np.where(outline)
        points = np.stack([m_y, m_x], axis=1)
        return points


class LabelImageTargetDextrDataset (DextrDataset):
    """
    A dataset of input images and corresponding label images.

    Each label image is an image with a 32-bit integer per pixel data type.
    The value 0 indicates background while non-zero identifies pixels belonging
    to the different objects in the image.
    `np.array(label_image)` should return an integer array.
    """
    def __init__(self, input_paths, label_image_paths, transform=None, obj_meta_path=None,
                 ignore_index=None, load_input=True, progress_fn=None):
        """
        Constructor

        :param input_paths: list of paths for input images
        :param label_image_paths: list of paths for label images, same length as `input_paths`
        :param transform: [optional] transformation to apply
        :param obj_meta_path: [optional] path for object cache file
        :param ignore_index: [optional] index of label to ignore
        :param load_input: if True, load input images, otherwise samples will not contain input images
        """
        self.input_paths = input_paths
        self.label_image_paths = label_image_paths
        self.IGNORE_INDEX = ignore_index

        super(LabelImageTargetDextrDataset, self).__init__(obj_meta_path, transform, load_input=load_input,
                                                           progress_fn=progress_fn)

    @property
    def num_images(self):
        return len(self.input_paths)

    def get_input_image_pil(self, img_i):
        path = self.input_paths[img_i]
        img = Image.open(path)
        img.load()
        return img

    def get_instance_y_arr(self, sample_i):
        path = self.label_image_paths[sample_i]
        img = Image.open(path)
        img.load()
        return np.array(img)


class MaskStackTargetDextrDataset (DextrDataset):
    """
    A dataset of input images and corresponding mask image stacks.

    Each mask stack is a list of binary images, one image per object. Most likely single channel uint8 type
    (PIL type 'L').
    """

    def __init__(self, input_paths, mask_stack_paths, transform=None, obj_meta_path=None,
                 ignore_index=None, load_input=True, progress_fn=None):
        """
        Constructor

        :param input_paths: list of paths for input images
        :param mask_stack_paths: nested list of paths for mask images: the input image at `input_paths[i]`
            should have a corresponding list of masks `mask_stack_paths[i]`, that lists the paths to the
            mask images, one mask per object. All the masks should have the same size as the corresponding
            input image.
        :param transform: [optional] transformation to apply
        :param obj_meta_path: [optional] path for object cache file
        :param ignore_index: [optional] index of label to ignore
        :param load_input: if True, load input images, otherwise samples will not contain input images
        """
        self.input_paths = input_paths
        self.mask_stack_paths = mask_stack_paths
        self.IGNORE_INDEX = ignore_index

        super(MaskStackTargetDextrDataset, self).__init__(obj_meta_path, transform, load_input=load_input,
                                                          progress_fn=progress_fn)

    @property
    def num_images(self):
        return len(self.input_paths)

    def get_input_image_pil(self, img_i):
        path = self.input_paths[img_i]
        img = Image.open(path)
        img.load()
        return img

    def get_instance_y_arr(self, sample_i):
        mask_paths = self.mask_stack_paths[sample_i]
        masks = []
        for p in mask_paths:
            img = Image.open(p)
            img.load()
            masks.append(np.array(img))
        return np.stack(masks, axis=0)
