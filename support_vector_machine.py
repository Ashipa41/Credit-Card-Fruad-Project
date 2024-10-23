# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('card_transdata.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target variable (fraud)

# Distribution of the Dependent Variable (Fraud vs Non-Fraud)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=dataset, x='fraud')
plt.title('Non-Fraudulent Transactions vs. Fraudulent')
plt.xlabel('FRAUD')
plt.ylabel('Count')

# Adding data labels to the bar plot
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),  # Offset points
                textcoords='offset points')
plt.show()

# Handling Class Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE
from collections import Counter

# Applying SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualize the distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Distribution of the Target Variable After SMOTE')
plt.xlabel('FRAUD')
plt.ylabel('Count')
plt.show()

# Checking class distribution before and after SMOTE
print(f'Class distribution before SMOTE: {Counter(y)}')
print(f'Class distribution after SMOTE: {Counter(y_resampled)}')

# Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=0)

# Training multiple SVM models with different kernels
from sklearn.svm import SVC

# Linear Kernel SVM
svm_model = SVC(kernel='linear', random_state=45)
svm_model.fit(X_train, y_train)

# Polynomial Kernel SVM
svm_model2 = SVC(kernel='poly', probability=True, random_state=45)
svm_model2.fit(X_train, y_train)

# Radial Basis Function (RBF) Kernel SVM
svm_model3 = SVC(kernel='rbf', probability=True, random_state=45)
svm_model3.fit(X_train, y_train)

# Predicting Test set results
# Polynomial Kernel Predictions
svm_pred2 = svm_model2.predict(X_test)
print(np.concatenate((svm_pred2.reshape(len(svm_pred2), 1), y_test.reshape(len(y_test), 1)), 1))

# Linear Kernel Predictions
svm_pred = svm_model.predict(X_test)
print(np.concatenate((svm_pred.reshape(len(svm_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# RBF Kernel Predictions
svm_pred3 = svm_model3.predict(X_test)
print(np.concatenate((svm_pred3.reshape(len(svm_pred3), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix and Accuracy for Polynomial Kernel
from sklearn.metrics import confusion_matrix, accuracy_score
cm2 = confusion_matrix(y_test, svm_pred2)
print("Confusion Matrix (Polynomial Kernel):\n", cm2)
print("Accuracy (Polynomial Kernel):", accuracy_score(y_test, svm_pred2))

# Confusion Matrix and Accuracy for Linear Kernel
cm = confusion_matrix(y_test, svm_pred)
print("Confusion Matrix (Linear Kernel):\n", cm)
print("Accuracy (Linear Kernel):", accuracy_score(y_test, svm_pred))

# Confusion Matrix and Accuracy for RBF Kernel
CM = confusion_matrix(y_test, svm_pred3)
print("Confusion Matrix (RBF Kernel):\n", CM)
print("Accuracy (RBF Kernel):", accuracy_score(y_test, svm_pred3))

# Display Confusion Matrix for Polynomial Kernel
from sklearn.metrics import ConfusionMatrixDisplay
conf_matrix_svm = confusion_matrix(y_test, svm_pred2)
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm, display_labels=[0, 1])
vis.plot(cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Support Vector Machine (Polynomial Kernel) Confusion Matrix')
plt.grid(False)
plt.show()

# Classification Report for Polynomial Kernel
from sklearn.metrics import classification_report
print("\nClassification Report (Polynomial Kernel):")
print(classification_report(y_test, svm_pred2))

# ROC Curve and AUC for Polynomial Kernel SVM
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

# Predicting probabilities for ROC-AUC
y_pred_svm_proba = svm_model2.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
print("AUC for Polynomial Kernel SVM:", auc_svm)

# Plotting the ROC curve
plt.plot(fpr_svm, tpr_svm, label="SVM (Polynomial Kernel), AUC={:.2f}".format(auc_svm))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Polynomial Kernel SVM')
plt.legend(loc=4)
plt.show()
