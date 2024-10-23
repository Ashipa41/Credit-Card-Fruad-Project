# Random Forest Classification with SMOTE (Synthetic Minority Over-sampling Technique)

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr

# Importing the dataset
dataset = pd.read_csv('card_transdata.csv')

# Separating features (X) and target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Dataset overview
print(dataset.shape)  # Shape of the dataset
print(dataset.info())  # Dataset information (columns, types)
print(dataset.head())  # Display first few rows
print(dataset.describe())  # Statistical summary of the dataset

# Checking for missing values
print("Missing values per column:")
print(dataset.isnull().sum())  # Sum of missing values per column

# Visualizing missing values with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='plasma')
plt.title('Missing Values Heatmap')
plt.show()

# Visualizing the distribution of numerical features using histograms
dataset.hist(bins=30, figsize=(20, 15))
plt.show()

# Visualizing data with boxplots to check for outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=dataset)
plt.show()

# Correlation testing using Spearman's correlation
spearman_corr_matrix = dataset.corr(method='spearman')
print("Spearman Correlation Matrix:\n", spearman_corr_matrix)

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Checking the proportion of the dependent variable ('fraud' column)
dependent_variable_counts = dataset['fraud'].value_counts()
print("Dependent Variable Distribution (fraud):\n", dependent_variable_counts)

# Visualizing the distribution of the dependent variable
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=dataset, x='fraud')
plt.title('Non-Fraudulent Transactions vs. Fraudulent')
plt.xlabel('FRAUD')
plt.ylabel('Count')

# Adding labels to the plot
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualizing the target variable distribution after applying SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Distribution of the Target Variable After SMOTE')
plt.xlabel('FRAUD')
plt.ylabel('Count')
plt.show()

# Displaying the class distribution before and after applying SMOTE
print(f'Class distribution before SMOTE: {Counter(y)}')
print(f'Class distribution after SMOTE: {Counter(y_resampled)}')

# Feature scaling using Min-Max Scaling
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Visualizing the scaled features using boxplots
plt.figure(figsize=(20, 10))
sns.boxplot(data=X_resampled)
plt.show()

# Splitting the dataset into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Displaying the shapes of the training and testing sets
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Labels Shape: {y_train.shape}")
print(f"Testing Labels Shape: {y_test.shape}")

# Training the Random Forest Classifier
classifier1 = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum depth of each tree
    max_features='sqrt',  # Number of features considered for each split
    min_samples_split=4,  # Minimum samples needed to split a node
    min_samples_leaf=2,   # Minimum samples needed for a leaf node
    bootstrap=True,       # Use bootstrap samples
    random_state=42
)
classifier1.fit(X_train, y_train)

# Predicting the test set results
y_pred_R1 = classifier1.predict(X_test)
print("Predicted vs Actual:\n", np.concatenate((y_pred_R1.reshape(len(y_pred_R1), 1), y_test.reshape(len(y_test), 1)), 1))

# Evaluating the model performance with confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred_R1)
print("Confusion Matrix:\n", cm)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_R1)}")

# Training another Random Forest model with different parameters
classifier = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=45)
classifier.fit(X_train, y_train)

# Predicting the test set results for the second model
y_pred_R = classifier.predict(X_test)
print("Predicted vs Actual (Second Model):\n", np.concatenate((y_pred_R.reshape(len(y_pred_R), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix and accuracy for the second model
cm = confusion_matrix(y_test, y_pred_R)
print("Confusion Matrix (Second Model):\n", cm)
print(f"Accuracy Score (Second Model): {accuracy_score(y_test, y_pred_R)}")

# Displaying the confusion matrix for visualization
conf_matrix = confusion_matrix(y_test, y_pred_R1)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1]).plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.grid(False)
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_R1))

# ROC-AUC curve and AUC score
y_pred_rf_proba = classifier.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred_R1)
auc_rf = metrics.roc_auc_score(y_test, y_pred_R1)
print(f"AUC RF: {auc_rf:.2f}")

# Plotting the ROC curve
plt.plot(fpr_rf, tpr_rf, label=f"RF, auc={auc_rf:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc=4)
plt.show()

# Feature importance in the Random Forest model
feature_importances = classifier.feature_importances_

# Creating a DataFrame to visualize feature importance
feature_names = dataset.columns[:-1]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()
plt.show()
