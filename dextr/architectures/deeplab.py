import torch, torch.nn as nn, torch.nn.functional as F
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

    return dict(model=model, pretrained_parameters=pretrained_parameters, new_parameters=new_parameters)

