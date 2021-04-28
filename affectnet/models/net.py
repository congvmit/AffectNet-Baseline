import torch
from torch import nn
import torch.nn.functional as F
from affectnet.models.backbones.vit import ViT
from affectnet.models.backbones.resnet import resnet50


class MLP(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)


class AFVit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViT(
            image_size=256,
            patch_size=32,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.mlp = MLP(dim=1024, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.mlp(x)


class AFResnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.mlp = MLP(dim=2048, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.mlp(x)


def get_model(model_name, *args, **kwargs):
    if model_name == 'vit':
        return AFVit(*args, **kwargs)

    elif model_name == 'resnet50':
        return AFResnet50(*args, **kwargs)