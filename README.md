# PyTorch implementation of DEXTR

An implementation of [DEXTR](http://people.ee.ethz.ch/~cvlsegmentation/dextr/).
The original implementation can be found at [https://github.com/scaelles/DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch).

This implementation is intended for use as a library.


## Training using the command line `train_dextr.py` program

### Train a DEXTR network using the Pascal VOC dataset

This will train a DEXTR model using a [U-Net](https://arxiv.org/abs/1505.04597) with a ResNet-101 based encoder.
It should take several hours on an nVidia 1080Ti GPU. 

- Download the Pascal VOC 2012 dataset [development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- Create a file called `dextr.cfg` with the following contents:
```cfg
[paths]
pascal_voc=<path to VOC2012 diretory>
```  
- Train the DEXTR model by running:

`> python train_dextr.py pascal_resunet101 --dataset=pascal_voc --arch=resunet101`

The name `pascal_resunet101` is the name of the job; STDOUT will be logged to `logs/log_pascal_resunet101.txt` and the model
file will be saved to `checkpoints/pascal_resunet101.pth`. You can give the job any name you like.


### Fine tuning a DEXTR network using a custom data set

There are two types of data set you can use:
1. Each input image has a corresponding label image, where label images have an integer pixel type such that each
   pixel gives the index of the object that covers it, or 0 for background. The Pascal VOC dataset is arranged in
   this way.
2. Each input image has a corresponding set of mask images that form a stack. Each mask image is an 8-bit greyscale
   image that corresponds to an object/instance and identifies the pixels covered by it.

Please arrange your custom data set so that the image file names (excluding extension) match or are a prefix
to the label/mask image file names. E.g. the image `img0.jpg` will match the label file `img0.png` or `img0_labels.png`.
For mask stack datasets `img0.jpg` would match to the mask images `img0_mask0.png`, ... `img0_maskN.png`.
The images and labels can live in separate directories; they are matched by filename *only*.

##### Training using a label image data set

`> python train_dextr.py my_model_from_labels --dataset=custom_label
--train_image_pat=/mydataset/train/input/*.jpg --train_target_pat=/mydataset/train/labels/*.png
--arch=resunet101 --load_model=checkpoints/pascal_resunet101.pth`

The input and label images are given to the `--train_image_pat` and `--train_target_pat` options.
You can specify validation images using the `--val_image_pat` and `--val_target_pat` options in a similar way.

`--load_model=checkpoints/pascal_resunet101.pth` indicates that we should start by loading the
model trained on Pascal VOC above and fine-tune it, rather than starting from an ImageNet classifier.

You can specify that the label index 255 should be ignore by adding `--label_ignore_index=255`.

You could train using the entire (train and validation) Pascal VOC data set using:

`> python train_dextr.py my_model_from_pascal --dataset=custom_label
--train_image_pat=/pascal/VOC2012/JPEGImages/*.jpg --train_target_pat=/pascal/VOC2012/SegmentationObjects/*.png
--label_ignore_index=255 --arch=resunet101`

##### Training using a mask stack data set

`> python train_dextr.py my_model_from_masks --dataset=custom_mask
--train_image_pat=/mydataset/train/input/*.jpg --train_target_pat=/mydataset/train/masks/*.png
--arch=resunet101 --load_model=checkpoints/pascal_resunet101.pth`



## Python Inference API

First, load the model using `torch.load`:

```py3
# Load model and switch to evaluation mode
MODEL_PATH = 'checkpoints/pascalvoc_resunet101.pth'
torch_device = torch.device('cuda:0')
dextr_model = torch.load(MODEL_PATH, map_location=torch_device)
dextr_model.eval()
```

The images that you use as input can take the form of either NumPy arrays or PIL Images. Each image should
have a corresponding list of four extreme points. The `DextrModel` class `predict` method takes a list
of `N` images and a `(N, 4, [y, x])` shaped NumPy array of extreme points. It returns a list of masks; each
mask is the same size as the corresponding input image:

```py3
masks = dextr_model.predict(images, extreme_points, torch_device)
```

See `demo.py` for an example of using the `dextr` inference API.


## Python training API

The `training_loop` function within the `dextr.model` module provides a simple training loop that can be used
for training or fine-tuning models. See `train_dextr.py` for usage.
