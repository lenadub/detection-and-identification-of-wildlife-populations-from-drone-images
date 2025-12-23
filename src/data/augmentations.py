# src/data/augmentations.py

import albumentations as A


def get_train_transforms():
    """
    Augmentations applied during training only.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.4
            ),
            A.MotionBlur(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"]
        )
    )


def get_val_transforms():
    """
    Validation / test transforms (no augmentation).
    """
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"]
        )
    )