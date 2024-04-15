import torch
from torchvision.models.resnet import ResNet


def resnet_forward_features(model: ResNet, x: torch.tensor) -> tuple[torch.tensor]:
    outs = []
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    outs.append(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    outs.append(x)
    x = model.layer2(x)
    outs.append(x)
    x = model.layer3(x)
    outs.append(x)
    x = model.layer4(x)
    outs.append(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)
    outs.append(x)

    return tuple(outs)
