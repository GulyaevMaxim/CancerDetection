import torchvision.models
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


def get_dense_net_169(pretrained=False):
    net = torchvision.models.densenet169()
    if pretrained:
        net = torchvision.models.densenet169(
            pretrained='imagenet')
    num_ftrs = net.classifier.in_features
    out_ftrs = int(net.classifier.out_features / 4)
    net.classifier = nn.Sequential(
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, out_ftrs, bias=True),
        nn.SELU(),
        nn.Dropout(0.7),
        nn.Linear(in_features=out_ftrs, out_features=1, bias=True),
    )

    return net


def get_dense_net_121(pretrained=False):
    net = torchvision.models.densenet121()
    if pretrained:
        net = torchvision.models.densenet121(
            pretrained='imagenet')
    num_ftrs = net.classifier.in_features
    out_ftrs = int(net.classifier.out_features / 4)
    net.classifier = nn.Sequential(
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, out_ftrs, bias=True),
        nn.SELU(),
        nn.Dropout(0.7),
        nn.Linear(in_features=out_ftrs, out_features=1, bias=True),
    )

    return net


def get_menet_456(pretrained=False):
    net = ptcv_get_model("menet456_24x1_g3", pretrained=False)
    if pretrained:
        net = ptcv_get_model("menet456_24x1_g3", pretrained=True)
    num_ftrs = net.output.in_features
    out_ftrs = int(net.output.out_features / 4)
    net.output = nn.Sequential(
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, out_ftrs, bias=True),
        nn.SELU(),
        nn.Dropout(0.7),
        nn.Linear(in_features=out_ftrs, out_features=1, bias=True),
    )
    return net


def get_resnet18(pretrained=False):
    net = torchvision.models.resnet18()
    if pretrained:
        net = torchvision.models.resnet18(
            pretrained='imagenet')
    num_ftrs = net.fc.in_features
    out_ftrs = int(net.fc.out_features / 4)
    net.fc = nn.Sequential(
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, out_ftrs, bias=True),
        nn.SELU(),
        nn.Dropout(0.7),
        nn.Linear(in_features=out_ftrs, out_features=1, bias=True),
    )
    return net


def get_resnet50(number_outputs, pretrained=False):
    net = torchvision.models.resnet50()
    if pretrained:
        net = torchvision.models.resnet50(
            pretrained='imagenet')
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, number_outputs)
    return net