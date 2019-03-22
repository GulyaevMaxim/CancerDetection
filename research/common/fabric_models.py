import torchvision.models
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


def get_dense_net_169(number_outputs, pretrained=False):
    net = torchvision.models.densenet169()
    if pretrained:
        net = torchvision.models.densenet169(
            pretrained='imagenet')
    num_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(num_ftrs, number_outputs)
    return net

def get_dense_net_121(number_outputs, pretrained=False):
    net = torchvision.models.densenet121()
    if pretrained:
        net = torchvision.models.densenet121(
            pretrained='imagenet')
    num_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(num_ftrs, number_outputs)
    return net

def get_menet_456(number_outputs, pretrained=False):
    net = ptcv_get_model("menet456_24x1_g3", pretrained=False)
    if pretrained:
        net = ptcv_get_model("menet456_24x1_g3", pretrained=True)
    num_ftrs = net.output.in_features
    net.output = nn.Linear(num_ftrs, number_outputs)
    return net


def get_resnet18(number_outputs, pretrained=False):
    net = torchvision.models.resnet18()
    if pretrained:
        net = torchvision.models.resnet18(
            pretrained='imagenet')
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, number_outputs)
    return net


def get_resnet50(number_outputs, pretrained=False):
    net = torchvision.models.resnet50()
    if pretrained:
        net = torchvision.models.resnet50(
            pretrained='imagenet')
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, number_outputs)
    return net
