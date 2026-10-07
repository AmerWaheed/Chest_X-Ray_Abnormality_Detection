# Chest X-Ray Abnormality Detection

## Overview

This project develops a **YOLOv8-based object detection model** for detecting and localizing abnormalities in chest X-ray images.

The goal is to explore how **transfer learning** can be used to automatically identify thoracic abnormalities and predict their locations using bounding boxes.

The project focuses on **object detection**, rather than simple image classification, meaning the model predicts:

* **What abnormality is present**
* **Where the abnormality is located**

---

## Problem Statement

Chest X-rays are widely used for detecting thoracic abnormalities. However, manually reviewing large numbers of radiographs can be time-consuming and may involve variability between readers.

This project investigates an automated computer vision approach that can detect and localize abnormalities in chest X-rays.

---

## Dataset

The project uses the **VinBigData Chest X-ray Abnormalities Detection dataset**, using a COCO-format version with WBF 3× downscaled images.

### Dataset Structure

* **3,296 training images**
* **1,098 validation images**
* **14 abnormality classes**
* Images contain one or more annotated abnormalities.
* Each abnormality is represented using a bounding box.

Examples of abnormalities include:

* Aortic enlargement
* Cardiomegaly
* Pneumothorax
* Nodule/Mass
* Pleural thickening
* Atelectasis
* Calcification
* Other lesions

The dataset is **multi-abnormality**, meaning a single X-ray can contain multiple different abnormality classes.

---

## Objective

The main objective is to train a YOLO-based object detector using **transfer learning from COCO-pretrained weights** and evaluate its ability to:

1. Detect abnormalities.
2. Localize abnormalities using bounding boxes.
3. Classify the detected abnormality.
4. Analyze performance across different abnormality classes.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* PIL
* OpenCV
* PyTorch
* Ultralytics YOLO
* YOLOv8
* COCO annotations
* Computer Vision
* Object Detection
* Transfer Learning

---

## Project Workflow

```text
VinBigData Dataset
        ↓
COCO Annotation Inspection
        ↓
Exploratory Data Analysis
        ↓
Class Distribution Analysis
        ↓
Bounding Box Analysis
        ↓
Annotation Validation
        ↓
COCO → YOLO Conversion
        ↓
YOLO Dataset Validation
        ↓
YOLOv8s Transfer Learning
        ↓
Model Training
        ↓
Validation & Evaluation
        ↓
Prediction Visualization
        ↓
Error Analysis
        ↓
Deployment Model Export
```

---

## 1. Dataset Analysis

The COCO annotations were inspected to verify:

* Image information
* Class IDs and names
* Bounding-box coordinates
* Number of annotations
* Training/validation consistency

The dataset was also analyzed to understand the distribution of abnormalities and the number of abnormalities appearing in each image.

---

## 2. Exploratory Data Analysis

The project investigates:

* Abnormality frequency by class
* Number of annotations per image
* Class co-occurrence
* Bounding-box width and height
* Bounding-box aspect ratios
* Relative bounding-box area
* Image dimensions

This analysis helps identify class imbalance and understand the visual characteristics of the detection problem.

---

## 3. Annotation Validation

Before training, the annotations were checked for:

* Negative or zero-area bounding boxes
* Out-of-bounds coordinates
* Duplicate annotations
* Images without annotations
* Invalid YOLO class IDs
* Invalid normalized coordinates

This step helps ensure that the training data is structurally valid.

---

## 4. COCO to YOLO Conversion

The original annotations were provided in **COCO format**.

They were converted into the YOLO format:

```text
class_id x_center y_center width height
```

The coordinates were normalized between `0` and `1`.

Separate YOLO datasets were created for:

```text
train/
val/
```

with corresponding image and label directories.

---

## 5. Model

A **YOLOv8s** model pretrained on the COCO dataset was used.

```python
model = YOLO("yolov8s.pt")
```

Transfer learning was used to adapt the pretrained model to the 14 chest X-ray abnormality classes.

### Training Configuration

* Model: YOLOv8s
* Epochs: 60
* Image size: 640 × 640
* Batch size: 16
* Seed: 42
* Early stopping patience: 15
* Pretrained COCO weights: Yes

---

## 6. Evaluation

The model was evaluated using object detection metrics including:

* Precision
* Recall
* mAP@50
* mAP@50–95

The final evaluation used:

* Confidence threshold: `0.25`
* IoU threshold: `0.50`

### Overall Results

| Metric         |    Result |
| -------------- | --------: |
| mAP@50         | **0.285** |
| mAP@50–95      | **0.151** |
| Mean Precision | **0.542** |
| Mean Recall    | **0.362** |

---

## 7. Per-Class Performance

Performance varied significantly between abnormality classes.

The model performed particularly well on:

| Class              |   mAP@50 |
| ------------------ | -------: |
| Cardiomegaly       | **0.85** |
| Aortic enlargement | **0.80** |

The weakest performance was observed for:

| Class              | mAP@50 |
| ------------------ | -----: |
| Calcification      |  0.075 |
| Other lesion       |  0.097 |
| Pleural thickening |  0.133 |

This shows that the number of training annotations is not the only factor affecting detection performance.

Large and clearly defined abnormalities were easier for the model to localize, while subtle, diffuse, or inconsistently bounded abnormalities were more difficult.

---

## 8. Prediction Visualization

The notebook includes visualization of model predictions on validation images.

The predictions display:

* Detected abnormality class
* Bounding box
* Model confidence

Ground-truth annotations were also visualized for comparison.

---

## 9. Error Analysis

Error analysis was performed using:

* Confusion matrix visualization
* Prediction visualization
* Low-confidence predictions
* Comparison between ground-truth annotations and model predictions
* Per-class precision and recall
* Per-class mAP

This helped identify which abnormalities were difficult for the model to detect.

---

## 10. Deployment Preparation

The trained model was exported into a deployment folder containing:

```text
deploy/
├── best.pt
└── classes.json
```

The `best.pt` file contains the trained YOLO model, while `classes.json` stores the class mapping required for inference.

---

## Key Findings

The project demonstrated that transfer learning with YOLOv8 can detect and localize several chest X-ray abnormalities.

However, performance was uneven across classes.

### Main observations

* Cardiomegaly achieved strong detection performance.
* Aortic enlargement also achieved strong performance.
* Small or subtle abnormalities were more difficult to detect.
* Rare classes generally suffered from limited training examples.
* High annotation frequency does not necessarily guarantee high detection performance.
* Bounding-box quality and visual characteristics of an abnormality are important factors.

---

## Limitations

This project used a single YOLOv8s configuration.

The following improvements were not explored:

* Hyperparameter optimization
* Comparison with larger YOLO architectures
* Advanced augmentation strategies
* Class-imbalance mitigation
* Oversampling
* Focal loss
* Extensive cross-validation
* Extensive threshold optimization

Therefore, the results should be considered a baseline rather than a production-ready medical diagnostic system.

---

## Future Improvements

Potential improvements include:

* Experimenting with YOLOv8m/l models
* Hyperparameter tuning
* Advanced image augmentation
* Addressing class imbalance
* Using focal loss or other imbalance strategies
* Increasing training epochs
* Testing different image resolutions
* Applying SHAP or additional interpretability methods where appropriate
* Building a web interface for inference
* Testing inference on external chest X-ray datasets
* Monitoring model performance after deployment

---

## Disclaimer

This project is an **educational and research-oriented computer vision project**.

The model is not intended to replace radiologists or provide medical diagnoses. Its predictions should not be used for clinical decision-making.

---

## Project Structure

```text
Chest-X-Ray-Abnormality-Detection/
│
├── cv-xray.ipynb
├── README.md
├── requirements.txt
│
├── model/
│   ├── best.pt
│   └── classes.json
│
└── images/
    ├── dataset_distribution.png
    ├── sample_annotations.png
    ├── predictions.png
    └── evaluation.png
```

---

## Conclusion

This project demonstrates an end-to-end **medical computer vision object detection pipeline**, starting from COCO-format annotations and exploratory analysis through YOLO conversion, transfer learning, model training, evaluation, visualization, error analysis, and deployment preparation.

The final YOLOv8s model achieved a **mAP@50 of 0.285** and demonstrated particularly strong performance on abnormalities such as **Cardiomegaly** and **Aortic enlargement**.

The project provided practical experience with **object detection, transfer learning, medical imaging, bounding-box annotation, YOLO, model evaluation, and error analysis**.
