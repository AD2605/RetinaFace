import torchvision

def resnet101(pretrained=True):
    return torchvision.models.resnet101(pretrained=pretrained)

def resnet50(pretrained=True):
    return torchvision.models.resnet50(pretrained=pretrained)

def resnext_50(pretrained=True):
    return torchvision.models.resnext50_32x4d(pretrained=pretrained)

def resnext_101(pretrained=True):
    return torchvision.models.resnext101_32x8d(pretrained=pretrained)
