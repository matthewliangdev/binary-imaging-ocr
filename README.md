# Handwritten Character Recognition | Binary Image Analysis

A Python computer vision project developed for an introductory Imaging and Multimedia course (CS 334) at Rutgers University. This project explores handwritten character recognition using classical image processing and nearest-neighbor classification instead of neural networks.

## Overview

The goal of this project is to identify handwritten characters through binary image processing, feature extraction, and classification. The system is trained on 800 handwritten character samples across 10 character classes.

## Methodology

1. **Image Thresholding:** Convert images to grayscale and compare fixed thresholds with Otsu thresholding.
2. **Connected Component Analysis:** Identify and isolate individual characters using 8-connected component labeling and bounding boxes.
3. **Feature Extraction:** Extract seven Hu moment features from each character and standardize them using training statistics.
4. **Classification:** Classify characters using Euclidean distance and 1-nearest neighbor.
5. **Evaluation:** Measure classification performance using confusion matrices and ground-truth comparisons.

## Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
binary-imaging-ocr/
├── README.md
├── .gitignore
├── requirements.txt
├── binary_imaging_ocr.ipynb
├── generate_demo_data.py
├── data/
├── docs/
└── results/
```

## Original Results

| Metric | Result |
|---|---:|
| Character classes | 10 |
| Training samples | 800 |
| Training recognition rate | 47.62% |
| Matched test characters | 70/70 |
| Correct test predictions | 37/70 |
| Test recognition rate | 52.86% |

The system successfully matched all 70 ground-truth character locations. However, classification accuracy was limited by similarities between handwritten character shapes and the limitations of the selected features.

The training evaluation uses the second sorted nearest neighbor to avoid self-matching, although distance ties may affect this assumption.

These results establish a baseline for future improvements.

## Installation and Usage

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Generate the Demo Dataset

```bash
python generate_demo_data.py
```

### 3. Configure the Notebook

Open `binary_imaging_ocr.ipynb` and set the dataset path in the first code cell:

```python
DATA_DIR = Path("data/demo")
```

### 4. Run the Notebook

```bash
python -m notebook
```

Open `binary_imaging_ocr.ipynb` and execute the cells in order.

The synthetic dataset uses printed characters, so its recognition results will differ from the original handwritten dataset results reported above.

## Dataset Availability

The original training images, test image, and ground-truth labels were provided as part of university coursework and are not distributed in this public repository.

A synthetic dataset generator is included to demonstrate the functionality of the image processing and classification pipeline independently of the original course materials.

Users with authorized access to the original dataset can place the files in the `data/` directory and set the notebook's dataset path to `Path("data")`. Only load trusted pickle files.

## Learning Objectives

This project demonstrates foundational experience with:

- Digital image processing
- Binary image analysis
- Image segmentation
- Feature extraction
- Dataset processing
- Classification algorithms
- Quantitative performance evaluation
- Data visualization

## Future Improvements

- Apply morphological operations to improve character segmentation.
- Explore additional shape descriptors to improve recognition accuracy.
- Compare different values of k in k-nearest-neighbor classification.
- Explicitly exclude self-matches during training evaluation.
- Evaluate performance on larger and more diverse datasets.

## Acknowledgments

Developed as part of an undergraduate Imaging and Multimedia course (CS 334) at Rutgers University to explore fundamental computer vision and image processing techniques.