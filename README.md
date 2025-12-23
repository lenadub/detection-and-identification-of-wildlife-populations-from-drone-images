# Wildlife Species Detection from Drone Imagery

## Project Overview

This project focuses on the design and implementation of an automated computer vision system for detecting and identifying wildlife species in aerial drone imagery.  
Using the **Wildlife Aerial Images from Drone (WAID)** dataset, the system detects multiple terrestrial species — **sheep, cattle, seal, camelus, kiang, and zebra** — using state-of-the-art deep learning–based object detection models.

The project compares several modern detection architectures:
- **YOLOv8** for fast, real-time detection
- **Faster R-CNN** for high-accuracy region-based detection
- **DETR (Transformer-based)** to evaluate attention mechanisms in complex aerial scenes

The final objective is a reproducible and efficient detection pipeline that can support wildlife monitoring and ecological research.

---

## Repository Structure

```text
wildlife_detection/
│── data/
│   ├── raw/                 # Raw dataset (NOT tracked by git)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── annotations/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── classes.txt          # Class names (tracked)
│   └── processed/           # Generated data (ignored by git)
│
│── notebooks/               # Jupyter notebooks (exploration & experiments)
│── src/                     # Core reusable source code
│── outputs/                 # Trained models, logs, visual results (ignored)
│── requirements.txt         # Python dependencies
│── README.md
````

---

## Dataset Setup (Required)

The **WAID dataset is not included** in this repository and must be downloaded manually from its official GitHub source:

➡️ **Dataset link:**
`https://github.com/xiaohuicui/WAID`

### After downloading the dataset:

1. Extract the dataset archive.

2. Move the image folders into:

   ```text
   data/raw/images/
   ```

3. Move the label folders into:

   ```text
   data/raw/annotations/
   ```

   > If the dataset provides a folder named `labels/`, **rename it to `annotations/`**.

4. Ensure the dataset follows this structure:

   ```text
   data/raw/images/train/
   data/raw/images/val/
   data/raw/images/test/

   data/raw/annotations/train/
   data/raw/annotations/val/
   data/raw/annotations/test/
   ```

---

## Class Labels

Class names are defined in:

```text
data/classes.txt
```

This file must contain **one class name per line**, in the exact order corresponding to YOLO class IDs:

```text
sheep
cattle
seal
kiang
camelus
zebra
```

**Important:**
The order of classes in `classes.txt` must match the class IDs used in the annotation files.
Do **not** change this order once training begins.

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

Activate the environment:

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

From the project root directory:

```bash
pip install -r requirements.txt
```

This installs all required libraries for data processing, training, and evaluation.

---

## Getting Started

Begin with the data exploration notebook:

```text
notebooks/data_exploration.ipynb
```

This notebook:

* Verifies dataset integrity
* Explores class distributions
* Analyzes bounding box sizes
* Visualizes representative samples

---

## Notes

* All experiments were conducted using **Python 3.9+**.
* Training deep learning models requires a **GPU** for reasonable performance.
* Raw data and generated outputs are intentionally excluded from version control to ensure reproducibility.

