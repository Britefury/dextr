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

import torch.nn as nn
from torchvision.models import segmentation

def dextr_deeplab3(num_classes=1):
    model = segmentation.deeplabv3_resnet101(pretrained=True)

    # Create new input `conv1` layer that takes a 4-channel input
    new_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_conv1.weight[:, :3, :, :] = model.backbone.conv1.weight
    model.backbone.conv1.weight.data = new_conv1.weight.data

    new_parameters = []

    # Create new classifier output layer
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_parameters.extend(list(model.classifier[-1].parameters()))

    if model.aux_classifier is not None:
        # Create new aux classifier output layer
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        new_parameters.extend(list(model.aux_classifier[-1].parameters()))

    # Setup new and pre-trained parameters for fine-tuning
    new_param_ids = {id(p) for p in new_parameters}
    pretrained_parameters = [p for p in model.parameters() if id(p) not in new_param_ids]

    model.pretrained_parameters = pretrained_parameters
    model.new_parameters = new_parameters

    return model

