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

import time
import math
import itertools
import numpy as np
from skimage.util import img_as_ubyte, img_as_float32
import PIL.Image
import torch, torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import Compose
from torch.hub import load_state_dict_from_url
from dextr.data_pipeline import dextr_transforms
from dextr import lr_schedules, repeat_sampler


class _DextrInferenceDataset (torch.utils.data.Dataset):
    def __init__(self, images, object_extreme_points, transform=None):
        if len(images) != len(object_extreme_points):
            raise ValueError('The number of images ({}) and the number of extreme point sets ({}) do not match'.format(
                len(images), len(object_extreme_points)
            ))
        self.images = images
        self.object_extreme_points = object_extreme_points
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        if isinstance(image, np.ndarray):
            image = img_as_ubyte(image)
            image = PIL.Image.fromarray(image)
        sample = dict(input=image, extreme_points=self.object_extreme_points[item])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class DextrModel (object):
    URL_PASCALVOC_RESUNET101 = 'https://storage.googleapis.com/dextr_pytorch_models_public/dextr_pascalvoc_resunet101-a2d81727.pth'

    def __init__(self, net, target_size_yx, padding, blob_sigma):
        """
        Constructor

        :param net: the DEXTR network; a PyTorch Module instance
        :param target_size_yx: the size to which crops are scaled for inference
        :param padding: amount of space between edges of crop and the closest extreme points
        :param blob_sigma: the sigma of the Gaussian blobs placed at each extreme point
        """
        self.net = net
        self.target_size_yx = target_size_yx
        self.padding = padding
        self.blob_sigma = blob_sigma

        self.__inference_transforms = Compose([
            dextr_transforms.DextrCropWithHeatmapTransform(
                target_size_yx=target_size_yx,
                padding=padding,
                blob_sigma=blob_sigma,
            ),
            dextr_transforms.DextrToTensor(),
            dextr_transforms.DextrNormalize(),
        ])

    def predict(self, images, object_extreme_points, torch_device, batch_size=4, num_workers=0):
        """
        Predict DEXTR masks for objects identified in images by extreme points.

        :param images: a list of N images; images can be PIL Images or NumPy arrays
        :param object_extreme_points: extreme points for each object/image as an array of `(N, 4, [y,x])` NumPy arrays
        :param torch_device: PyTorch device used
        :param batch_size: batch size used (relevant when using a large number of images)
        :param num_workers: number of background processes used by the data pipeline
        :return: mask for each image in images, where each mask is a NumPy array
        """
        ds = _DextrInferenceDataset(images, object_extreme_points, transform=self.__inference_transforms)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        sample_i = 0
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input = batch['input'].to(torch_device)
                crop_yx = batch['crop_yx'].detach().cpu().numpy()

                pred_logits = self.net(input)['out']

                pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()

                for i in range(len(crop_yx)):
                    image_size = images[sample_i].size[::-1]
                    pred_pil = dextr_transforms.paste_mask_into_image(
                        image_size, pred_prob[i, 0, :, :], crop_yx[i])

                    pred_pil_arr = img_as_float32(np.array(pred_pil))

                    predictions.append(pred_pil_arr)

                    sample_i += 1

        return predictions

    def train(self):
        """Switch to training mode.
        """
        self.net = self.net.train()

    def eval(self):
        """Switch to evaluation mode.
        """
        self.net = self.net.eval()

    def to(self, device):
        """Move network to specified device.
        """
        self.net = self.net.to(device)
        return self

    @staticmethod
    def pascalvoc_resunet101(map_location='cpu'):
        return load_state_dict_from_url(DextrModel.URL_PASCALVOC_RESUNET101,
                                        map_location=map_location, check_hash=True)


def training_loop(model, train_ds_fn, val_ds_fn, val_truth_ds_fn, extreme_range, noise_std,
                  learning_rate, pretrained_lr_factor, new_lr_factor,
                  lr_sched, lr_poly_power, opt_type, sgd_weight_decay, aug_hflip, aug_rot_range,
                  batch_size, iters_per_epoch, num_epochs, val_every_n_epochs,
                  torch_device, num_workers, verbose, on_epoch_finished=None):
    """Train a DEXTR model.

    :param model: the model to train as a `DextrModel` instance
    :param train_ds_fn: function that returns the training set, of the form `fn(transform=None)`. Returns an
        instance of a subclass of `torch.utils.data.Dataset`. The `transform` parameter would normally be passed to
        the dataset constructor.
    :param val_ds_fn: function that returns the validation set, see `train_ds_fn`
    :param val_truth_ds_fn: function that returns the validation set, without transforms, where each sample only needs
        to contain the target object mask image
    :param extreme_range: when the object mask is scaled to match the size of the target crop used for
        training/inference, extreme points will be selected from mask pixels that are within this range
        of the edge (5 pixels is most likely a good choice)
    :param noise_std: standard deviation of Gaussian noise added to extreme point positions during training
        (1.0 would be a good choice)
    :param learning_rate: learning rate used during training (0.1 works well for SGD)
    :param pretrained_lr_factor: learning rate scaling factor used for fine-tuning pre-trained parameters (0.1)
    :param new_lr_factor: learning rate scaling factor used for training new parameters. Use 1.0 if starting
        from an ImageNet pre-trained classifier with new output layers. If starting from a trained DEXTR
        network use 0.1.
    :param lr_sched: learning rate schedule type 'none' for constant schedule, 'cosine' for a cosine schedule
        that uses a cosine wave to scale the learning rate to 0 over the course of training, or 'poly' for
        a polynomial schedule, or a callable/function of the from `fn(optimizer, T_max)` where `T_max` is
        the total number of iterations
    :param lr_poly_power: when `lr_sched == 'poly'` this value determines the power; the learning rate used is
        `learning_rate * (1 - current_iter / total_iters) ** lr_poly_power`
    :param opt_type: optimizer type 'sgd', 'adam' or a callable of the form `fn(param_groups)`
    :param sgd_weight_decay: the weight decay value used when using an SGD optimizer
    :param aug_hflip: if True, use random horizontal flip augmentation
    :param aug_rot_range: rotation range in degrees, e.g. a value of 20 will rotate samples
        between -20 and 20 degrees
    :param batch_size: the batch size for training
    :param iters_per_epoch: the number of iterations per epoch
    :param num_epochs: the number of epochs to train for
    :param val_every_n_epochs: the number of epochs between each validation
    :param torch_device: the torch device used for training
    :param num_workers: the number of worker processes used to build data batches
    :param verbose: if True, print progress using `print` function
    :param on_epoch_finished: [optional] callback invoked after finishing each epoch
    :return:
    """
    train_transform = Compose([
        dextr_transforms.DextrTrainingTransform(
            target_size_yx=model.target_size_yx,
            padding=model.padding,
            extreme_range=extreme_range,
            noise_std=noise_std,
            blob_sigma=model.blob_sigma,
            hflip=aug_hflip,
            rot_range=math.radians(aug_rot_range),
            fix_box_to_extreme=True
        ),
        dextr_transforms.DextrToTensor(),
        dextr_transforms.DextrNormalize(),
    ])

    eval_transform = Compose([
        dextr_transforms.DextrFindExtremesTransform(
            target_size_yx=model.target_size_yx,
            extreme_range=extreme_range,
        ),
        dextr_transforms.DextrCropWithHeatmapTransform(
            target_size_yx=model.target_size_yx,
            padding=model.padding,
            blob_sigma=model.blob_sigma,
        ),
        dextr_transforms.DextrToTensor(),
        dextr_transforms.DextrNormalize(),
    ])

    # Build dataset
    train_ds = train_ds_fn(transform=train_transform)
    if val_ds_fn is not None and val_truth_ds_fn is not None:
        val_ds = val_ds_fn(transform=eval_transform)
        val_truth_ds = val_truth_ds_fn()
    else:
        val_ds = None
        val_truth_ds = None

    model = model.to(torch_device)

    # Create optimizer
    if opt_type == 'sgd':
        optimizer = torch.optim.SGD([
            dict(params=model.net.pretrained_parameters, lr=learning_rate * pretrained_lr_factor),
            dict(params=model.net.new_parameters, lr=learning_rate * new_lr_factor)],
            momentum=0.9, nesterov=True, weight_decay=sgd_weight_decay, lr=learning_rate)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.net.pretrained_parameters, lr=learning_rate * pretrained_lr_factor),
            dict(params=model.net.new_parameters, lr=learning_rate * new_lr_factor)])
    elif callable(opt_type):
        optimizer = opt_type([
            dict(params=model.net.pretrained_parameters, lr=learning_rate * pretrained_lr_factor),
            dict(params=model.net.new_parameters, lr=learning_rate * new_lr_factor)])
    else:
        print('Unknown optimizer type {}'.format(opt_type))
        return

    # LR schedule
    if iters_per_epoch == -1:
        iters_per_epoch = len(train_ds) // batch_size
    total_iters = iters_per_epoch * num_epochs

    lr_iter_scheduler = lr_schedules.make_lr_scheduler(
        optimizer=optimizer, total_iters=total_iters, schedule_type=lr_sched,
        poly_power=lr_poly_power
    )


    train_sampler = repeat_sampler.RepeatSampler(torch.utils.data.RandomSampler(train_ds))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, sampler=train_sampler,
                                               num_workers=num_workers)

    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size, num_workers=num_workers)
    else:
        val_loader = None


    train_iter = iter(train_loader)

    if verbose:
        # Report dataset size
        print('Dataset:')
        print('len(train_ds)={}'.format(len(train_ds)))
        if val_ds is not None:
            print('len(val_ds)={}'.format(len(val_ds)))

    def validate():
        val_loss_accum = 0.0
        val_iou_accum = 0.0
        n_val_batches = 0
        n_ious = 0
        val_i = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input = batch['input'].to(torch_device)
                target_mask = batch['target_mask'].to(torch_device)
                crop_yx = batch['crop_yx'].detach().cpu().numpy()

                # Compute class balance
                target_mask_flat = target_mask.view(len(target_mask), -1)
                target_balance = target_mask_flat.mean(dim=1)
                target_balance = target_balance[:, None, None, None]
                bg_weight = target_balance
                fg_weight = 1.0 - target_balance
                weight = bg_weight + (fg_weight - bg_weight) * target_mask

                pred_out = model.net(input)
                pred_logits = pred_out['out']

                loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, weight=weight)
                val_loss_accum += float(loss)

                pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()

                for i in range(len(crop_yx)):
                    true_mask = np.array(val_truth_ds[val_i]['target_mask']) > 127
                    pred_mask_pil = dextr_transforms.paste_mask_into_image(
                        true_mask.shape, pred_prob[i, 0, :, :], crop_yx[i])
                    pred_mask = np.array(pred_mask_pil) > 127

                    iou_denom = (pred_mask | true_mask).sum()
                    if iou_denom > 0.0:
                        iou = (pred_mask & true_mask).sum() / iou_denom

                        val_iou_accum += iou
                        n_ious += 1

                    val_i += 1


                n_val_batches += 1

        return (val_loss_accum / n_val_batches, val_iou_accum / n_ious)



    if verbose:
        print('Training...')
    iter_i = 0
    validated = False
    for epoch in range(num_epochs):
        t1 = time.time()

        train_loss_accum = 0.0

        model.train()

        for batch in itertools.islice(train_iter, iters_per_epoch):
            if lr_iter_scheduler is not None:
                lr_iter_scheduler.step(iter_i)

            input = batch['input'].to(torch_device)
            target_mask = batch['target_mask'].to(torch_device)

            optimizer.zero_grad()

            # Compute class balance
            target_mask_flat = target_mask.view(len(target_mask), -1)
            target_balance = target_mask_flat.mean(dim=1)
            target_balance = target_balance[:, None, None, None]
            bg_weight = target_balance
            fg_weight = 1.0 - target_balance
            weight = bg_weight + (fg_weight - bg_weight) * target_mask

            pred_out = model.net(input)
            pred_logits = pred_out['out']

            loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, weight=weight)

            loss.backward()

            optimizer.step()

            train_loss_accum += float(loss)

            iter_i += 1

        train_loss_accum /= iters_per_epoch

        if verbose:
            if val_ds is not None and (epoch + 1) % val_every_n_epochs == 0:
                val_loss, val_iou = validate()
                validated = True
            else:
                val_loss = val_iou = None
                validated = False

            t2 = time.time()

            if val_loss is not None:
                print('Epoch {} took {:.3f}s: loss={:.6f}, VAL loss={:.6f}, mIoU={:.3%}'.format(
                    epoch + 1, t2 - t1, train_loss_accum, val_loss, val_iou))
            else:
                print('Epoch {} took {:.3f}s: loss={:.6f}'.format(epoch + 1, t2 - t1, train_loss_accum))


        if on_epoch_finished is not None:
            on_epoch_finished(epoch, model)

    if verbose:
        if not validated and val_ds is not None:
            val_loss, val_iou = validate()
            print('FINAL: VAL loss={:.6f}, mIoU={:.3%}'.format(val_loss, val_iou))
