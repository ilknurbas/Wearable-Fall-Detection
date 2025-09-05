### Fall Detection with Wearable Sensor Data

This project implements a fall detection pipeline using wearable sensor data (566×306 dataset) to classify fall versus non-fall actions. Both unsupervised and supervised methods were applied, including PCA for dimensionality reduction, K-Means clustering for exploratory analysis, and supervised classifiers such as Multi-layer Perceptron (MLP) and Support Vector Machines (SVM) for accurate classification.


**`Key Steps & Techniques:`**

+ **Dataset:** 566 samples × 306 features (wearable sensor readings) 

+ **Dimensionality Reduction:** PCA (Principal Component Analysis)  to extract top principal components for visualization and preprocessing

+ **Clustering:** K-Means with different cluster numbers for exploratory analysis

+ **Classification:**

    - MLP (Multi-layer Perceptron)
    
    - SVM (Support Vector Machine)

+ **Evaluation:** Accuracy on validation and test sets

+ **Tools:**  Python, NumPy, scikit-learn, Matplotlib

**`Highlights:`**

+ Achieved consistent high classification accuracy across MLP and SVM

+ Visualized sensor data clusters with PCA for exploratory insights

+ Demonstrated the effectiveness of PCA in preprocessing high-dimensional sensor data

**`Usage:`**

```
# Install dependencies
pip install numpy scikit-learn matplotlib

# Run the main script
python fall_detection.py
```
