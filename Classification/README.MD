# Combined Classification Models for Acne Severity Detection and Image Classification

This repository contains three different image classification and detection models, each solving unique challenges related to acne detection and general image classification. The models leverage state-of-the-art techniques such as **YOLO**, **Vision Transformer (ViT)**, and **Dlib with OpenCV** to process, analyze, and classify images effectively.

---

## Table of Contents
1. [Acne Severity Detection using YOLO and OpenCV](#acne-severity-detection-using-yolo-and-opencv)
2. [Vision Transformer Classification Model](#vision-transformer-classification-model)
3. [Acne Detection using Dlib and OpenCV](#acne-detection-using-dlib-and-opencv)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [License](#license)

---

## 1. Acne Severity Detection using YOLO and OpenCV
### Overview  
This project utilizes the **YOLOv8** object detection model to detect acne spots on facial images and compute acne severity based on the number and size of detected spots.

### Key Features  
- **YOLO Integration:** Uses the YOLOv8-xs model for acne spot detection.  
- **Preprocessing:** Image resizing and normalization for enhanced accuracy.  
- **Visualization:** Loss graphs, bounding boxes for acne regions, and severity scoring.  

### Workflow  
1. **Data Preprocessing:** Resize and normalize images for YOLO compatibility.  
2. **Model Training:** Train YOLO on an acne detection dataset while monitoring loss curves.  
3. **Inference:** Visualize acne detection results and compute severity scores.

### Dependencies  
- YOLO (Ultralytics)  
- OpenCV  
- Matplotlib  
- Pandas  

### Dataset  
- Roboflow Acne Detection Dataset  

---

## 2. Vision Transformer Classification Model
### Overview  
This project implements a **Vision Transformer (ViT-B/16)** model for image classification tasks, replacing traditional CNNs with transformer-based architecture for sequence-like image processing.

### Key Features  
- **Model:** Vision Transformer Base (ViT-B/16).  
- **Training Pipeline:** Image resizing (224x224), normalization, and tensor conversion.  
- **Metrics:** Accuracy, Cross-Entropy Loss, and performance graphs.  
- **Visualization:** Training loss curves and a confusion matrix for evaluation.  

### Workflow  
1. **Dataset Preparation:** Input images are resized, normalized, and processed into tensors.  
2. **Training:** Optimize model using CrossEntropyLoss and save checkpoints for the best-performing model.  
3. **Inference:** Use a softmax output to predict class probabilities.

### Dependencies  
- PyTorch  
- Torchvision  
- PIL  
- Matplotlib  

---

## 3. Acne Detection using Dlib and OpenCV
### Overview  
This project employs **Dlib** for face and landmark detection along with **OpenCV** for acne spot analysis. Severity is calculated based on the relative area affected by acne spots.

### Key Features  
- **Face and Landmark Detection:** Leverages Dlib's pre-trained shape predictor.  
- **Region Isolation:** Extract specific facial regions for focused acne analysis.  
- **Acne Detection:** Contour detection using thresholding techniques.  
- **Severity Scoring:** Quantify severity based on acne area percentage.  

### Workflow  
1. **Face Detection:** Identify facial landmarks.  
2. **Region Extraction:** Isolate specific areas (e.g., cheeks, forehead).  
3. **Acne Detection:** Apply image processing techniques to detect acne spots.  
4. **Severity Calculation:** Compute severity scores based on acne regions.

### Dependencies  
- Dlib  
- OpenCV  
- Matplotlib  

---

## Installation
Install the required dependencies for all three projects:

```bash
pip install opencv-python dlib numpy matplotlib torch torchvision ultralytics pandas
```

---

## How to Run

### 1. YOLO Acne Detection
```bash
git clone https://github.com/ahmedansari9548/SkinSync/tClassification/Yolo_v8_Acne_Detection
cd SkinSync/tClassification/Yolo_v8_Acne_Detection
jupyter notebook acne-Severity_YOLO.ipynb
```

### 2. Vision Transformer Model
```bash
git clone https://github.com/ahmedansari9548/SkinSync/Classification/Base_Model
cd SkinSync/Classification/Base_Model
jupyter notebook Classification_Base_Model.ipynb
```

### 3. Dlib Acne Detection
```bash
git clone https://github.com/ahmedansari9548/SkinSync/Classification/AcneSeverity.git
cd SkinSync/Classification/AcneSeverity
jupyter notebook AcneSeverity_Main.ipynb
```

---

## Results

### YOLO Model Results  
- **Train Loss:** 6.0  
- **Validation Loss:** 7.0  
- **Box Loss:** 4.0  
- **Class Loss:** 2.5  
- **Detection Accuracy:** 92%  

### Vision Transformer Results  
- **Best Loss:** 0.546  
- **Best Accuracy:** 92.3%  

### Dlib Acne Detection  
- **Output:** Processed images with facial regions, acne spots, and severity scores.  

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

--- 

### Contributors  
Your contributions are welcome! Submit a pull request or open an issue for improvements.

---

