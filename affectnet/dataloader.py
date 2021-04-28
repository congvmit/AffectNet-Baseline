import cv2
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
import os
import random

class AffectNetDataloader(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, 
                 test_dataset=None, 
                 num_workers=1, 
                 batch_size=32, 
                 pin_memory=True):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          worker_init_fn=self.seed_worker,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          worker_init_fn=self.seed_worker,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          worker_init_fn=self.seed_worker,
                          pin_memory=self.pin_memory)


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class AffectNetDataset(Dataset):
    def __init__(self, data_dir: str,
                 csv_path: str,
                 split: str,
                 transform: transforms.Compose = None):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.transform = transform
        self.split = split        


    def __getitem__(self, idx):
        # 'subDirectory_filePath', 'face_x', 'face_y',
        # 'face_width', 'face_height', 'facial_landmarks',
        # 'expression', 'valence', 'arousal'
        data = self.data_df.iloc[idx]

        image_path = data.subDirectory_filePath
        image_path = os.path.join(self.data_dir, image_path)

        expression = data.expression
        expression = torch.as_tensor(expression, dtype=torch.long)

        img_arr = cv2.imread(image_path)[..., ::-1]

        img_h, img_w, img_c = img_arr.shape

        # Crop face
        x = int(data.face_x)
        y = int(data.face_y)
        w = int(data.face_width)
        h = int(data.face_height)

        face_arr = img_arr[y:y + h, x:x + w, ...]

        if self.transform:
            face_arr = self.transform(image=face_arr)['image']

        # Load valence, arousal
        # -1 1
        valence = torch.as_tensor([data.valence], dtype=torch.float32)
        arousal = torch.as_tensor([data.arousal], dtype=torch.float32)

        # Load landmarks
        # landmarks = np.array(
        #     list(map(float, data.facial_landmarks.split(';'))))
        # landmarks = landmarks.reshape((68, 2))
        # landmarks[:, 0] /= img_w
        # landmarks[:, 1] /= img_h
        # landmarks = torch.from_numpy(landmarks.flatten()).float()

        return {
            'image': face_arr,  # img should be a tensor
            'expression': expression,
            'valence': valence,
            'arousal': arousal,
            # 'landmarks': landmarks
        }

    def __len__(self):
        return len(self.data_df)
