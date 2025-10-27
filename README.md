# Categorization of Indus Valley Inscriptions Using Machine Learning Algorithms

## Overview
This project applies Machine Learning and Computer Vision techniques to categorize inscriptions from the Indus Valley Civilization. It uses Histogram of Oriented Gradients (HOG) for feature extraction, Principal Component Analysis (PCA) for dimensionality reduction, and K-Means clustering for symbol grouping. The results contribute to computational approaches for interpreting ancient writing systems.

## Methodology

### 1. Data Preprocessing
The dataset contained 1,840 images of 70×54 pixels. Each image was normalized and denoised for better clarity.

### 2. Feature Extraction
Features were extracted using Histogram of Oriented Gradients (HOG) to capture edge orientation and symbol structure.

### 3. Dimensionality Reduction
Principal Component Analysis (PCA) was used to reduce feature dimensionality and visualize the patterns efficiently.

### 4. Clustering
K-Means clustering was applied to group symbols with similar characteristics. The Elbow Method determined the optimal number of clusters, and Silhouette Score was used for validation.


## Technology Stack
- Programming Language: Python  
- Libraries: NumPy, Pandas, OpenCV, scikit-learn, Matplotlib, Graphviz  
- Algorithms: HOG, PCA, K-Means  

## Future Work
1. Extend the model using deep learning (CNNs) for supervised symbol classification.  
2. Integrate transfer learning approaches such as ResNet or VGG.  
3. Develop a web-based visualization interface for interactive analysis.

## Citation
Rebekah Russel, Anshara Beigh, Dr. Shajee Mohan  
*Categorization of Indus Valley Inscriptions Using ML Algorithms*  
Department of Computer Science and Engineering, Sharda University, India (2025)
