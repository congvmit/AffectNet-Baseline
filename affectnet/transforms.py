import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform

import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch

import numpy as np
from PIL import Image
import cv2


def square_pad_torch(img):
    h, w, c = img.shape
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)

    img = Image.fromarray(img)  # Because torch only supports PIL images
    img = F.pad(img, padding, 0, 'constant')
    return img


def square_pad_cv2(img):
    h, w = img.shape[:2]
    max_wh = np.max([w, h])

    diff_vert = max_wh - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top

    diff_hori = max_wh - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left

    img_padded = cv2.copyMakeBorder(img,
                                    pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=0)
    return img_padded


class SquarePad(BasicTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError(
                "Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        img = square_pad_cv2(img)
        return img

    def get_transform_init_args_names(self):
        return ()


class Normalize(BasicTransform):
    """Convert image and mask to `torch.Tensor` and divide by 255 if image or mask are `uint8` type."""

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError(
                "Albumentations only supports images in HW or HWC format")
        if torch.is_tensor(img):
            img = img.type(torch.float32)
            if len(img.shape) == 2:
                img = img.unsqueeze(2)
        else:
            img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)

        img /= 255.0
        return img

    def get_transform_init_args_names(self):
        return ()


# ===============================================================================
# EMOTIC Transformations
# ===============================================================================
SCALE_LIMIT = 0.05
SHIFT_LIMIT = 0.05
ROTATE_LIMIT = 10

# ===============================================================================
# Transformations
# ===============================================================================
# For training

def get_train_transform(image_size):
    return A.Compose([
        # A.RandomCrop(256, 256),
        # A.CenterCrop(224, 224),
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=SHIFT_LIMIT, scale_limit=SCALE_LIMIT,
                        rotate_limit=ROTATE_LIMIT, p=.75, value=0,
                        border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
