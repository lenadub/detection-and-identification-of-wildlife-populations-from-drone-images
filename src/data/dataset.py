# src/data/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_image
from .utils import load_yolo_annotations


class WAIDDataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_dir,
        image_files,
        num_classes,
        transforms=None,
        image_size=(640, 640)
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = image_files
        self.num_classes = num_classes
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(
            self.annotation_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, labels = load_yolo_annotations(
            ann_path,
            self.num_classes
        )

        if self.transforms:
            augmented = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["class_labels"]

        image = preprocess_image(image, self.image_size)

        return {
            "image": torch.tensor(image).permute(2, 0, 1),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }