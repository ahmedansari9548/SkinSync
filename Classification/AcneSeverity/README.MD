
# Skin Acne Detection using Dlib and OpenCV

## Overview
This project focuses on detecting and analyzing acne severity in facial images using computer vision techniques. The system leverages **Dlib** for facial landmark detection and **OpenCV** for isolating regions of interest on the face.

The main objective is to compute an acne severity score by analyzing the relative area affected by acne spots.

---

## Features
- **Facial Detection**: Detect human faces and key facial landmarks using Dlib's pre-trained models.
- **Region Isolation**: Extract specific facial regions for acne analysis.
- **Acne Spot Detection**: Identify contours on the skin that represent acne.
- **Severity Scoring**: Calculate the severity of acne based on the relative area of affected regions.

---

## Tech Stack
- **Python**
- **Dlib** (for facial landmark detection)
- **OpenCV** (for image processing)
- **Matplotlib** (for visualizing results)

---

## Installation
Ensure you have the following libraries installed:

```bash
pip install opencv-python dlib numpy matplotlib
```

### Dependencies
- Python 3.x
- OpenCV
- Dlib (shape predictor model)

---

## Usage
1. Clone this repository:

```bash
git clone https://https://github.com/ahmedansari9548/SkinSync/Classification/AcneSeverity.git
cd acne-detection
```

2. Place your facial images in the `data/` folder.
3. Run the Jupyter Notebook:

```bash
jupyter notebook AcneSeverity_Main.ipynb
```

4. Follow the instructions in the notebook to process images and compute acne severity scores.

---

## Workflow
1. **Face Detection**: Uses Dlib's frontal face detector to locate the face in the input image.
2. **Facial Landmark Detection**: Identifies key points on the face (eyes, nose, mouth, cheeks, etc.).
3. **Region Isolation**: Extracts facial regions for focused analysis.
4. **Acne Spot Detection**:
   - Applies image processing techniques (thresholding, contour detection) to isolate acne spots.
5. **Severity Calculation**: Computes acne severity by analyzing the total area of acne spots relative to the skin region.

---

## Results
The notebook provides:
- **Processed Images** with facial regions and acne spots highlighted.
- **Severity Score** indicating the extent of acne.

Example output:
![Sample Output](images/sample_output.png)

---

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---
