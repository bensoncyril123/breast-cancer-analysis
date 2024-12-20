# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data.csv')

# Data Cleaning
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # Drop unnecessary columns
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode diagnosis as numeric

# Checking for missing values
if df.isnull().sum().sum() == 0:
    print("No missing values in the dataset.")

# EDA: Basic Statistics and Distribution Visualizations
print("Dataset shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nDescriptive Statistics:\n", df.describe())

# Plot histograms for all features
df.hist(bins=20, figsize=(20, 15))
plt.show()

# Visualizing the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Feature Selection: High Correlation Features
correlation_threshold = 0.75
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation_features = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
print("\nHighly Correlated Features:\n", high_correlation_features)

# Data Splitting
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.0 Classification Models

# K-Nearest Neighbors (KNN)
error_rate = []
for k in range(1, 42):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rate.append(np.mean(pred_k != y_test))

optimal_k = error_rate.index(min(error_rate)) + 1
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Logistic Regression
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=400, random_state=42)
random_forest.fit(X_train_scaled, y_train)
y_pred_rf = random_forest.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Support Vector Machine (SVM) with RBF Kernel
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test_scaled)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

# 5.0 Results Summary

# Accuracy Scores
models = [
    "K-Nearest Neighbors (KNN)",
    "Logistic Regression",
    "Random Forest",
    "SVM with RBF Kernel",
    "Gradient Boosting"
]

accuracies = [
    accuracy_knn,
    accuracy_log_reg,
    accuracy_rf,
    accuracy_svm_rbf,
    accuracy_gb
]

# Display and Plot Accuracy Scores
results_df = pd.DataFrame({"Model": models, "Accuracy": accuracies}).sort_values(by="Accuracy", ascending=False)
print("\nModel Performance:\n", results_df)

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.barh(results_df["Model"], results_df["Accuracy"], color='skyblue')
plt.title('Model Accuracies')
plt.xlabel('Accuracy')
plt.show()
