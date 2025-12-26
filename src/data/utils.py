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

def filter_invalid_yolo_bboxes(bboxes, labels, eps=1e-6):
    """
    Remove YOLO bboxes with zero or negative area.
    """
    new_bboxes = []
    new_labels = []

    for bbox, label in zip(bboxes, labels):
        x, y, w, h = bbox
        if w > eps and h > eps:
            new_bboxes.append(bbox)
            new_labels.append(label)

    return new_bboxes, new_labels

def sanitize_yolo_bboxes(bboxes, labels, eps=1e-6):
    """
    Keep only valid YOLO bboxes:
    - width > 0 and height > 0
    - all values finite
    - (optionally) x,y within [0,1] and w,h within (0,1]
    """
    clean_bboxes, clean_labels = [], []
    for bb, lab in zip(bboxes, labels):
        x, y, w, h = bb

        # basic numeric checks
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        if not (w > eps and h > eps):
            continue
        if not (w <= 1 and h <= 1):
            continue

        clean_bboxes.append([x, y, w, h])
        clean_labels.append(lab)

    return clean_bboxes, clean_labels