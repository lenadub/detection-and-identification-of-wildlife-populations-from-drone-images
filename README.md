# Wildlife Species Detection from Drone Imagery

## Project Overview

This project focuses on the design and implementation of an automated computer vision system for detecting and identifying wildlife species in aerial drone imagery.  
Using the **Wildlife Aerial Images from Drone (WAID)** dataset, the system detects multiple terrestrial species — **sheep, cattle, seal, kiang, camelus, and zebra** — using state-of-the-art deep learning–based object detection models.

The project investigates and compares multiple detection paradigms:
- **YOLOv8** for fast, real-time detection
- **Faster R-CNN** for high-accuracy region-based detection
- **DETR (Transformer-based)** to assess attention-based, set-prediction models

The final objective is a **reproducible, modular detection pipeline** suitable for wildlife monitoring and ecological research.

---

## Repository Structure

```text
detection-and-identification-of-wildlife-populations-from-drone-images/
│
│── data/
│   ├── raw/                         # Raw WAID dataset (NOT tracked)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── annotations/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   │
│   ├── classes.txt                  # Class names (tracked)
│   └── processed/                   # Optional generated data
│
│── notebooks/
│   ├── data_exploration.ipynb
│   ├── data_preprocessing.ipynb
│   ├── evaluation_and_results.ipynb
│   ├── model_training_detr.ipynb
│   ├── model_training_faster_rcnn.ipynb
│   ├── model_training_yolo.ipynb
│   └── outputs/
│       └── faster_rcnn_waid.pth          # Trained Faster
│
│── src/
│   └── data/
│       ├── __init__.py
│       ├── dataset.py               # PyTorch Dataset (WAIDDataset)
│       ├── preprocessing.py         # Resizing & normalization
│       ├── augmentations.py         # Albumentations pipelines
│       └── utils.py                 # Annotation & bbox utilities
│
│── requirements.txt
│── README.md
````

---

## Dataset Setup (Required)

The **WAID dataset is not included** in this repository and must be downloaded manually from its official source:

➡️ **Dataset link:**
[https://github.com/xiaohuicui/WAID](https://github.com/xiaohuicui/WAID)

### After downloading:

1. Extract the dataset.
2. Place images into:

   ```text
   data/raw/images/
   ```
3. Place labels into:

   ```text
   data/raw/annotations/
   ```

   > If the dataset provides a folder named `labels/`, **rename it to `annotations/`**.
4. Ensure the final structure is:

   ```text
   data/raw/images/train/
   data/raw/images/valid/
   data/raw/images/test/

   data/raw/annotations/train/
   data/raw/annotations/valid/
   data/raw/annotations/test/
   ```

---

## Class Labels

Class names are defined in:

```text
data/classes.txt
```

Content (one class per line, order matters):

```text
sheep
cattle
seal
kiang
camelus
zebra
```

⚠️ **Important:**
The order of `classes.txt` must exactly match YOLO class indices used in annotation files.
Changing this order will invalidate training and evaluation.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wildlife_detection
```

---

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

* **Windows**

  ```bash
  venv\Scripts\activate
  ```

* **macOS / Linux**

  ```bash
  source venv/bin/activate
  ```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Workflow Overview

### 1️⃣ Data Exploration

Notebook:

```text
notebooks/data_exploration.ipynb
```

This notebook:

* Verifies dataset integrity
* Explores class imbalance
* Analyzes bounding box sizes
* Visualizes representative samples

---

### 2️⃣ Preprocessing Pipeline

Notebook:

```text
notebooks/data_preprocessing.ipynb
```

Implementation:

* Core preprocessing functions are defined in:

  ```text
  src/data/preprocessing.py
  ```
* Augmentations are defined in:

  ```text
  src/data/augmentations.py
  ```

Preprocessing includes:

* Image resizing to a fixed resolution
* Pixel-value normalization
* Optional data augmentation (training only)

These steps are applied automatically through the dataset class:

```python
from src.data.dataset import WAIDDataset
```

This design ensures consistent preprocessing across training, validation, and testing.

---

### 3️⃣ Faster R-CNN Training

Notebook:

```text
notebooks/model_training_faster_rcnn.ipynb
```

Key characteristics:

* Uses `torchvision`’s Faster R-CNN with a ResNet-FPN backbone
* Fine-tuned using transfer learning
* Supports class-aware sampling to mitigate class imbalance
* Validation loss monitored during training

The trained model is saved as:

```text
outputs/faster_rcnn_waid.pth
```

The trained model can be loaded for evaluation using:

```python
model.load_state_dict(torch.load("outputs/faster_rcnn_waid.pth"))
model.eval()
```

---

## Notes

* All experiments were conducted using **Python 3.9+**. 
* Training deep learning models requires a **GPU** for reasonable performance. 
* Raw data and generated outputs are intentionally excluded from version control to ensure reproducibility.
