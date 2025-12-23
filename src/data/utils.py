# src/data/utils.py

import os


def load_classes(classes_path):
    """
    Load class names from classes.txt
    """
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def load_yolo_annotations(annotation_path, num_classes):
    """
    Load YOLO annotations from a file.

    Returns:
        bboxes (list): [x, y, w, h]
        labels (list): class ids
    """
    bboxes = []
    labels = []

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cid, x, y, w, h = map(float, parts)
            cid = int(cid)

            if cid < 0 or cid >= num_classes:
                raise ValueError(
                    f"Invalid class id {cid} in {annotation_path}"
                )

            bboxes.append([x, y, w, h])
            labels.append(cid)

    return bboxes, labels