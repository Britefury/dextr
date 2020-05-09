import math
import numpy as np
import PIL.Image
import torch
from dextr.data_pipeline import affine

_SQRT_2_PI = math.sqrt(2.0 * math.pi)


def _choose_item(items, rng):
    """
    Randomly choose an item from the array items

    :param items: items as a NumPy array of shape `(N, ...)`
    :param rng: RNG used for choosing item
    :return: an item from `items`
    """
    if len(items) == 1:
        return items[0]
    else:
        i = rng.randint(0, len(items))
        return items[i]

def _select_extreme(mask, extreme_range, rng):
    """
    Randomly select a pixel in the mask that has a non-zero value and return its co-ordinates,
    with an offset added.

    :param mask: mask as a NumPy array of shape `(H, W)`
    :param extreme_range: range to randomly search
    :return: `(y, x)` co-ordinates of the chosen pixel, with offset applied
    """
    m_y, m_x = np.where(mask)
    m_yx = np.stack([m_y, m_x], axis=1)

    top_filter = m_y <= (m_y.min() + extreme_range)
    bottom_filter = m_y >= (m_y.max() - extreme_range)
    left_filter = m_x <= (m_x.min() + extreme_range)
    right_filter = m_x >= (m_x.max() - extreme_range)
    top_extreme = _choose_item(m_yx[top_filter], rng)
    bottom_extreme = _choose_item(m_yx[bottom_filter], rng)
    left_extreme = _choose_item(m_yx[left_filter], rng)
    right_extreme = _choose_item(m_yx[right_filter], rng)

    return np.stack([top_extreme, bottom_extreme, left_extreme, right_extreme], axis=0)


def _gaussian_kernels(size, sigma, centres):
    """
    Compute gaussian kernels in PyTorch

    :param size: size of the kernel 'image'
    :param sigma: Gaussian sigma
    :param centres: the location of the centres of the blobs as a `(N,)` Torch tensor
    :param torch_device: Torch device to use

    :return: NumPy array tensor of shape `(N, size,)`
    """
    x = np.arange(size, dtype=float)
    return np.exp(-0.5 * (x - centres[..., None]) ** 2 / sigma ** 2)


class DextrTrainingTransform (object):
    def __init__(self, target_size_yx, padding, extreme_range, noise_std=1.0, blob_sigma=10.0,
                 hflip=True, rot_range=math.radians(20.0), fix_box_to_extreme=True, rng=None):
        self.target_size_yx = target_size_yx
        self.hflip = hflip
        self.rot_range = rot_range
        self.padding = padding
        self.extreme_range = extreme_range
        self.noise_std = noise_std
        self.blob_sigma = blob_sigma
        self.target_size_xy = np.array(target_size_yx[::-1]).astype(float)
        self.target_size_nopad_xy = self.target_size_xy - self.padding * 2
        self.fix_box_to_extreme = fix_box_to_extreme
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    def __call__(self, sample):
        mask_pil = sample['target_mask']
        mask = np.array(mask_pil)
        outline_yx = sample['target_mask_outline'] + 0.5
        image_size = mask.shape[:2]

        rng = self.rng

        # Horizontal flip
        flip_flags = (rng.binomial(1, 0.5, size=(1, 3)) * np.array([1, 0, 0])) != 0
        flip_xf = affine.flip_xyd_matrices(flip_flags, image_size)

        # Rotation
        theta = rng.uniform(-self.rot_range, self.rot_range)
        rot_xf = affine.rotation_matrices([theta])

        # Apply flip and rotation to hull points
        rot_flip_xf = affine.cat_nx2x3(rot_xf, flip_xf)
        outline_rotflip_xy = affine.transform_points(rot_flip_xf[0], outline_yx[:, ::-1])
        outline_rotflip_xy_min = np.floor(outline_rotflip_xy.min(axis=0)).astype(int)
        outline_rotflip_xy_max = np.ceil(outline_rotflip_xy.max(axis=0)).astype(int)
        outline_rotflip_xy_size = outline_rotflip_xy_max - outline_rotflip_xy_min

        if self.fix_box_to_extreme:
            # Scale to target size
            scale = self.target_size_xy / outline_rotflip_xy_size.astype(float)
            scale_xf = affine.scale_matrices(scale[None, :])

            # Compose transformation from:
            # flip, rotate, top-left corner, scale
            extreme_xf = affine.cat_nx2x3(
                scale_xf,
                affine.translation_matrices(-outline_rotflip_xy_min[None, :]),
                rot_xf, flip_xf)
            extreme_xf_pil = affine.inv_nx2x3(extreme_xf)

            # Transform mask
            mask_extreme_pil = mask_pil.transform(
                self.target_size_yx[::-1], PIL.Image.AFFINE, tuple(extreme_xf_pil.flatten().tolist()),
                resample=PIL.Image.NEAREST)
            mask_extreme = np.array(mask_extreme_pil) > 127

            # Select extreme points
            extreme_points = _select_extreme(mask_extreme, self.extreme_range, rng)
            extreme_points = extreme_points + rng.normal(size=extreme_points.shape) * self.noise_std

            # Compute the bounding box given the points
            points_lower_yx = extreme_points.min(axis=0)
            points_upper_yx = extreme_points.max(axis=0)
            points_size_yx = points_upper_yx - points_lower_yx

            # Scale points so that their bounds fit the target size minus padding
            points_frac_yx = (extreme_points - points_lower_yx) / np.maximum(points_size_yx, 1.0)
            final_points_yx = points_frac_yx * self.target_size_nopad_xy[::-1] + self.padding

            # Final DEXTR transform:
            # - existing extreme points transform
            # - move top left corner of extreme points to [0,0]
            # - scale bounds of extreme points to the target size minus padding
            # - padding (translation)
            points_scale_xy = self.target_size_nopad_xy / np.maximum(points_size_yx, 1.0)[::-1]
            dextr_xf = affine.cat_nx2x3(
                affine.translation_matrices(np.array([[self.padding, self.padding]])),
                affine.scale_matrices(points_scale_xy[None, ...]),
                affine.translation_matrices(-points_lower_yx[::-1][None, ...]),
                extreme_xf,
            )
            dextr_xf_pil = affine.inv_nx2x3(dextr_xf)

            # Create blobs
            gauss_y = _gaussian_kernels(self.target_size_yx[0], self.blob_sigma, final_points_yx[:, 0])
            gauss_x = _gaussian_kernels(self.target_size_yx[1], self.blob_sigma, final_points_yx[:, 1])
            heatmap = (gauss_y[:, :, None] * gauss_x[:, None, :]).max(axis=0)

            input_dextr_pil = sample['input'].transform(
                self.target_size_yx[::-1], PIL.Image.AFFINE, tuple(dextr_xf_pil.flatten().tolist()),
                resample=PIL.Image.BILINEAR)
            mask_dextr_pil = mask_pil.transform(
                self.target_size_yx[::-1], PIL.Image.AFFINE, tuple(dextr_xf_pil.flatten().tolist()),
                resample=PIL.Image.BILINEAR)

            return dict(input=input_dextr_pil, target_mask=mask_dextr_pil, heatmap=heatmap)
        else:
            # Scale to target size, accounting for padding
            scale = self.target_size_nopad_xy / outline_rotflip_xy_size.astype(float)
            scale_xf = affine.scale_matrices(scale[None, :])

            # Compose transformation from:
            # flip, rotate, top-left corner, scale, pad
            dextr_xf = affine.cat_nx2x3(
                affine.translation_matrices(np.array([[self.padding, self.padding]])),
                scale_xf,
                affine.translation_matrices(-outline_rotflip_xy_min[None, :]),
                rot_xf, flip_xf)
            dextr_xf_pil = affine.inv_nx2x3(dextr_xf)

            # Transform mask
            mask_dextr_pil = mask_pil.transform(
                self.target_size_yx[::-1], PIL.Image.AFFINE, tuple(dextr_xf_pil.flatten().tolist()),
                resample=PIL.Image.BILINEAR)
            mask_extreme = np.array(mask_dextr_pil) > 127

            # Select extreme points, within
            extreme_points = _select_extreme(mask_extreme, self.extreme_range, rng)
            extreme_points = extreme_points + rng.normal(size=extreme_points.shape) * self.noise_std

            # Create blobs
            gauss_y = _gaussian_kernels(self.target_size_yx[0], self.blob_sigma, extreme_points[:, 0])
            gauss_x = _gaussian_kernels(self.target_size_yx[1], self.blob_sigma, extreme_points[:, 1])
            heatmap = (gauss_y[:, :, None] * gauss_x[:, None, :]).max(axis=0)

            input_dextr_pil = sample['input'].transform(
                self.target_size_yx[::-1], PIL.Image.AFFINE, tuple(dextr_xf_pil.flatten().tolist()),
                resample=PIL.Image.BILINEAR)

            return dict(input=input_dextr_pil, target_mask=mask_dextr_pil, heatmap=heatmap)


class DextrFindExtremesTransform (object):
    def __init__(self, target_size_yx, extreme_range, rng=None):
        self.target_size_yx = target_size_yx
        self.extreme_range = extreme_range
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    def __call__(self, sample):
        outline_yx = sample['target_mask_outline'] + 0.5

        rng = self.rng

        # Apply flip and rotation to hull points
        outline_yx_min = np.floor(outline_yx.min(axis=0))
        outline_yx_max = np.ceil(outline_yx.max(axis=0))

        # Transform mask
        mask_extreme_pil = sample['target_mask'].transform(
            self.target_size_yx[::-1], method=PIL.Image.EXTENT,
            data=(outline_yx_min[1], outline_yx_min[0], outline_yx_max[1], outline_yx_max[0]),
            resample=PIL.Image.NEAREST
        )
        mask_extreme = np.array(mask_extreme_pil) > 127

        # Select extreme points
        extreme_points_in_tgt = _select_extreme(mask_extreme, self.extreme_range, rng)
        extreme_points_frac = extreme_points_in_tgt / np.array(self.target_size_yx)

        extreme_points = outline_yx_min + extreme_points_frac * (outline_yx_max - outline_yx_min)

        return dict(input=sample['input'], target_mask=sample['target_mask'],
                    target_mask_outline=sample['target_mask_outline'], extreme_points=extreme_points)



def crop_with_heatmap(input_image, target_mask, target_size_yx, padding, extreme_points, blob_sigma):
    """
    Crop the input image (and optionally target mask) to a region surrounding the extreme points
    and create a corresponding heat map with gaussian blobs centred on the extreme points.

    :param input_image: input image as a `PIL.Image`
    :param target_mask: [optional] target mask as a `PIL.Image`
    :param target_size_yx: target crop size
    :param padding: padding in target crop space that lies between the extreme points and the closest
        edge of the target crop
    :param extreme_points: extreme points as a `(4, [y, x])` NumPy array
    :param blob_sigma: Gaussian blob sigma size
    :return: `(cropped_input, cropped_mask, heatmap, crop_yx)` where:
        cropped_input is the crop taken from the input image, scaled to `target_size_yx`, as a PIL.Image
        cropped_mask is the crop taken from the target mask image, scaled to `target_size_yx` as a PIL.Image,
            or None if target_mask is None
        heatmap is the heat map as a 2D NumPy array
        crop_yx the crop region taken from the input and mask images as a `[[lower_y, lower_x], [upper_y, upper_x]]`
            NumPy array
    """
    target_size_nopad_yx = np.array(target_size_yx) - padding * 2
    rel_padding_yx = padding / target_size_nopad_yx

    # Compute the bounding box given the points
    points_lower_yx = extreme_points.min(axis=0)
    points_upper_yx = extreme_points.max(axis=0)
    points_size_yx = points_upper_yx - points_lower_yx
    crop_lower_yx = points_lower_yx - points_size_yx * rel_padding_yx
    crop_upper_yx = points_upper_yx + points_size_yx * rel_padding_yx
    crop_size_yx = crop_upper_yx - crop_lower_yx

    # Scale points so that their bounds fit the target size minus padding
    points_frac_yx = (extreme_points - crop_lower_yx) / np.maximum(crop_size_yx, 1.0)
    points_tgt_yx = points_frac_yx * np.array(target_size_yx)

    # Create blobs
    gauss_y = _gaussian_kernels(target_size_yx[0], blob_sigma, points_tgt_yx[:, 0])
    gauss_x = _gaussian_kernels(target_size_yx[1], blob_sigma, points_tgt_yx[:, 1])
    heatmap = (gauss_y[:, :, None] * gauss_x[:, None, :]).sum(axis=0)

    input_dextr_pil = input_image.transform(
        target_size_yx[::-1], method=PIL.Image.EXTENT,
        data=(int(crop_lower_yx[1]), int(crop_lower_yx[0]), int(crop_upper_yx[1]), int(crop_upper_yx[0])),
        resample=PIL.Image.BILINEAR)

    crop_yx = np.stack([crop_lower_yx, crop_upper_yx], axis=0)

    if target_mask is not None:
        mask_dextr_pil = target_mask.transform(
            target_size_yx[::-1], method=PIL.Image.EXTENT,
            data=(crop_lower_yx[1], crop_lower_yx[0], crop_upper_yx[1], crop_upper_yx[0]),
            resample=PIL.Image.BILINEAR)
    else:
        mask_dextr_pil = None

    return input_dextr_pil, mask_dextr_pil, heatmap, crop_yx


class DextrCropWithHeatmapTransform (object):
    def __init__(self, target_size_yx, padding, blob_sigma=10.0, keep_original_mask=False):
        self.padding = padding
        self.blob_sigma = blob_sigma
        self.keep_original_mask = keep_original_mask
        self.target_size_yx = target_size_yx
        self.target_size_nopad_yx = np.array(target_size_yx) - self.padding * 2
        self._rel_padding_yx = padding / self.target_size_nopad_yx

    def __call__(self, sample):
        input_dextr_pil, mask_dextr_pil, heatmap, crop_yx = crop_with_heatmap(
            sample['input'], sample.get('target_mask'), self.target_size_yx, self.padding,
            sample['extreme_points'], self.blob_sigma
        )

        out_sample = dict(input=input_dextr_pil, heatmap=heatmap, crop_yx=crop_yx)

        if 'target_mask' in sample:
            out_sample['target_mask'] = mask_dextr_pil
        return out_sample


class DextrToTensor (object):
    def __call__(self, sample):
        input_pil = sample['input']
        heatmap = sample['heatmap']

        input_rgb = np.array(input_pil).astype(np.float32) / 255.0
        input_rgbh = np.append(input_rgb, heatmap[:, :, None], axis=2)

        out_sample = dict(
            input=torch.tensor(input_rgbh.transpose(2, 0, 1), dtype=torch.float)
        )

        if 'target_mask' in sample:
            mask = sample['target_mask']
            mask = np.array(mask).astype(np.float32) / 255.0
            out_sample['target_mask'] = torch.tensor(mask[None, :, :], dtype=torch.float)

        if 'crop_yx' in sample:
            out_sample['crop_yx'] = torch.tensor(sample['crop_yx'], dtype=torch.float)

        return out_sample


class DextrNormalize (object):
    """
    We need per-channel mean and standard devidation.
    Channels are R, G, B, H where H is the heat map channel.

    For RGB, use the ImageNet mean and std-dev specified in torchvision:
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    For X, computed from Pascal VOC 2012:
    mean=0.008, std=0.066

    So:
    >>> DextrNormalize(mean=[0.485, 0.456, 0.406, 0.008],
    ...                std=[0.229, 0.224, 0.225, 0.066])

    Or simply:
    >>> DextrNormalize()
    to use the above default values.
    """
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406, 0.008]
        if std is None:
            std = [0.229, 0.224, 0.225, 0.066]
        if len(mean) != 4:
            raise ValueError('mean should have 4 channels, not {}'.format(len(mean)))
        if len(std) != 4:
            raise ValueError('stdev should have 4 channels, not {}'.format(len(std)))
        mean = np.array(mean)
        std = np.array(std)
        self.mean = torch.tensor(mean[:, None, None], dtype=torch.float)
        self.std = torch.tensor(std[:, None, None], dtype=torch.float)

    def __call__(self, sample):
        out_sample = sample.copy()
        out_sample['input'] = (sample['input'] - self.mean) / self.std
        return out_sample


def paste_mask_into_image(image_size, mask_arr, target_crop):
    """

    :param image_size: output image size as a `(H, W)` tuple
    :param mask_arr: predicted mask as a `(target_h, target_w)` NumPy array
    :param target_crop: image space box that mask should be pasted (with scaling) into
    :return: PIL.Image of type 'L' (uint8, single channel) of size `image_size[::-1]`
    """
    mask_size_yx = np.array(mask_arr.shape[:2])

    crop_lower_yx = target_crop[0]
    crop_upper_yx = target_crop[1]
    crop_size_yx = crop_upper_yx - crop_lower_yx

    crop_lower_px_yx = np.floor(crop_lower_yx).astype(int)
    crop_upper_px_yx = np.ceil(crop_upper_yx).astype(int)

    crop_lower_px_yx = np.maximum(crop_lower_px_yx, 0)
    crop_upper_px_yx = np.minimum(crop_upper_px_yx, np.array(image_size))

    src_mask_crop_lower_yx = mask_size_yx * (crop_lower_px_yx - crop_lower_yx) / crop_size_yx
    src_mask_crop_upper_yx = mask_size_yx * (crop_upper_px_yx - crop_lower_yx) / crop_size_yx

    output_size = crop_upper_px_yx - crop_lower_px_yx

    if (output_size > 0).all():
        if issubclass(mask_arr.dtype.type, np.floating):
            mask_arr = (mask_arr * 255.0).astype(np.uint8)

        mask_pil = PIL.Image.fromarray(mask_arr)

        crop_pil = mask_pil.transform(
            tuple(output_size)[::-1], method=PIL.Image.EXTENT,
            data=(src_mask_crop_lower_yx[1], src_mask_crop_lower_yx[0],
                  src_mask_crop_upper_yx[1], src_mask_crop_upper_yx[0]),
            resample=PIL.Image.BILINEAR)

        out_pil = PIL.Image.new('L', image_size[::-1], 0)
        out_pil.paste(crop_pil, (crop_lower_px_yx[1], crop_lower_px_yx[0]))

        return out_pil
    else:
        return PIL.Image.new('L', image_size[::-1], 0)

