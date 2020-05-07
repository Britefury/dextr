"""
Pascal VOC dataset

Place VOC2012 dataset in 'VOC2012' directory.
For training, you will need the augmented labels. Download http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip.
The folder structure should be like:
VOC2012/JPEGImages
       /SegmentationClassAug
"""
import os, pickle
import tqdm
import numpy as np
import settings
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from PIL import Image
import torch.utils.data


class DextrDataset (torch.utils.data.Dataset):
    IGNORE_INDEX = None

    def __init__(self, object_meta_path, transform):
        if os.path.exists(object_meta_path):
            obj_meta = pickle.load(open(object_meta_path, 'rb'))
            self.obj_meta_indices = obj_meta['indices']
            self.obj_meta_outlines = obj_meta['outlines']
        else:
            self.obj_meta_indices, self.obj_meta_outlines = self._build_object_metadata()
            obj_meta = dict(indices=self.obj_meta_indices, outlines=self.obj_meta_outlines)
            pickle.dump(obj_meta, open(object_meta_path, 'wb'))

        self.transform = transform


    def __len__(self):
        return len(self.obj_meta_indices)

    def __getitem__(self, item):
        # Get object metadata
        img_i, label_i, region_i = self.obj_meta_indices[item]
        outline = self.obj_meta_outlines[item]

        # Get input image
        input_pil = self.get_input_image_pil(img_i)

        # Get the label mask and select the region
        label_mask = self.get_label_mask(img_i, label_i)
        if region_i != -1:
            regions, num_regions = label(label_mask)
            object_mask = regions == (region_i + 1)
        else:
            object_mask = label_mask
        object_mask_u8 = object_mask.astype(np.uint8) * 255
        object_mask_pil = Image.fromarray(object_mask_u8)

        sample = dict(input=input_pil, target_mask=object_mask_pil, target_mask_outline=outline)

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

    def _build_object_metadata(self):
        obj_meta_indices = []
        obj_meta_outlines = []

        for img_i in tqdm.tqdm(range(self.num_images), desc='Building object list'):
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
        interior = binary_erosion(mask)
        if interior.sum() < 1:
            return None
        else:
            outline = mask & ~binary_erosion(mask)
            m_y, m_x = np.where(outline)
            points = np.stack([m_y, m_x], axis=1)
            return points
