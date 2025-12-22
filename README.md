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
│── data/                # Raw and processed datasets
│── notebooks/           # Jupyter notebooks for exploration and experiments
│── src/                 # Core reusable source code
│── outputs/             # Trained models, logs, and visual results
│── requirements.txt     # Python dependencies
│── README.md
````

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wildlife_detection
```

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

From the project root directory, install all required Python packages:

```bash
pip install -r requirements.txt
```

This ensures that all libraries required for data processing, model training, and evaluation are available.

---

## Getting Started

Begin with the data exploration notebook:

```text
notebooks/data_exploration.ipynb
```

This notebook provides an overview of the WAID dataset, including image characteristics, class distribution, and annotation quality.

---

## Notes

* All experiments were conducted using Python 3.9+.
* Training deep learning models requires a GPU for reasonable performance.
