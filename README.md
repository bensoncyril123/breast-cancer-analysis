# Predictive Modeling and Data Analysis of Breast Cancer

## Overview

This project focuses on using machine learning techniques to predict the malignancy of breast cancer cases based on data derived from Fine Needle Aspiration (FNA) of breast masses. 
By analyzing cell characteristics, the project aims to enhance breast cancer diagnostics, enabling healthcare professionals to make more accurate and timely decisions. 
The study involves five classification models to evaluate their efficacy in distinguishing between malignant and benign cases.

---

## Objectives

1. Improve breast cancer diagnostics by utilizing machine learning techniques.
2. Predict the malignancy of breast cancer cases using cell characteristics extracted from FNA images.
3. Evaluate the performance of multiple machine learning models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - Support Vector Machines (SVM) with RBF Kernel
   - Gradient Boosting

---

## Dataset Description

The dataset contains features derived from digitized images of FNA samples. These features describe the characteristics of cell nuclei and are used as predictors for malignancy. The key attributes include:

- **Diagnosis**: The target variable (Malignant or Benign).
- **Features**: Mean, standard error, and "worst" measurements of:
  - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension.

### Data Summary

- **Total Samples**: 569
- **Class Distribution**:
  - Benign: 357
  - Malignant: 212
- **Number of Features**: 30 (after preprocessing).

---

## Methodology

1. **Data Preprocessing**:
   - Handled missing values and irrelevant features (e.g., ID and unnamed columns).
   - Scaled numerical features for model compatibility.
   - Split data into training (80%) and testing (20%) sets.

2. **Exploratory Data Analysis**:
   - Visualized feature distributions and correlations.
   - Analyzed relationships between features and target diagnosis.

3. **Model Building**:
   - Implemented the following machine learning models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Random Forest
     - SVM with RBF Kernel
     - Gradient Boosting
   - Tuned hyperparameters and evaluated performance metrics.

4. **Evaluation Metrics**:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - ROC Curve
   - Precision-Recall Curve

---

## Results

### Model Performance

| Model                          | Accuracy  |
|--------------------------------|-----------|
| **Support Vector Machines (SVM)** | **98.25%** |
| Logistic Regression            | 97.37%    |
| K-Nearest Neighbors (KNN)      | 96.49%    |
| Random Forest                  | 96.49%    |
| Gradient Boosting              | 95.61%    |

### Key Findings

- **SVM with RBF kernel** achieved the highest accuracy, making it the most effective model in predicting malignancy.
- Logistic Regression and KNN followed closely, offering reliable accuracy and robust diagnostics.
- Gradient Boosting and Random Forests also provided valuable insights with strong accuracy scores.

---

## Visualizations

1. **Countplot** of Diagnoses: Showed a higher prevalence of benign cases in the dataset.
2. **Feature Correlation Heatmap**: Highlighted relationships between features (e.g., strong correlation between `radius_mean`, `perimeter_mean`, and `area_mean`).
3. **ROC Curves**: Demonstrated the classification ability of each model.

---

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Machine Learning Frameworks**: scikit-learn

---

## Conclusion

The project demonstrated the effectiveness of machine learning in improving breast cancer diagnostics. 
SVM with RBF Kernel emerged as the best-performing model, offering an accuracy of 98.25%. 
These findings highlight the potential for integrating machine learning into clinical workflows to aid in early detection and improved patient outcomes.

---

## Future Work

1. Experiment with additional datasets to validate findings.
2. Explore deep learning approaches for image-based breast cancer detection.
3. Develop an interactive diagnostic tool for healthcare professionals.

---

## Author

- **Benson Cyril Nana Boakye**

For any inquiries, feel free to reach out via [nanaboab@gmail.com].
