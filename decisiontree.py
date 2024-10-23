# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve

# Step 1: Load the Dataset
# The dataset is assumed to be a CSV file containing transaction data with the last column as the target label (fraud).
Data_set = pd.read_csv('card_transdata.csv')
X = Data_set.iloc[:, :-1].values  # Features (excluding the target)
y = Data_set.iloc[:, -1].values   # Target (fraud label)

# Step 2: Inspect the Dataset
# Get an overview of the dataset structure, missing values, and statistics.
Data_set.info()  # Summary of dataset (types, non-null counts)
Data_set.head()  # Display first few rows of the dataset
Data_set.describe()  # Descriptive statistics for numerical features

# Step 3: Check for Missing Values
# Check and visualize any missing values in the dataset using a heatmap.
print(Data_set.isnull().sum())  # Print the count of missing values per column

plt.figure(figsize=(12, 8))
sns.heatmap(Data_set.isnull(), cbar=False, cmap='plasma')  # Heatmap for missing values
plt.title('Missing Values Heatmap')
plt.show()

# Step 4: Visualize Data Distribution
# Histograms for numerical features to understand their distribution.
Data_set.hist(bins=30, figsize=(20, 15))  # Histograms for all features
plt.show()

# Boxplots to visualize the spread and detect potential outliers.
plt.figure(figsize=(20, 10))
sns.boxplot(data=Data_set)  # Box plots for numerical features
plt.show()

# Step 5: Distribution of the Target Variable (Fraud vs Non-Fraud)
# Visualize the class distribution of the target (fraud).
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=Data_set, x='fraud')
plt.title('Non-Fraudulent Transactions vs. Fraudulent')
plt.xlabel('Fraud Label')
plt.ylabel('Count')

# Add count labels to the bar plot
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()

# Step 6: Handle Class Imbalance using SMOTE
# SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance in the dataset.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)  # Resampling the dataset

# Visualize the distribution after applying SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Distribution of Fraud after SMOTE')
plt.xlabel('Fraud Label')
plt.ylabel('Count')
plt.show()

# Print class distribution before and after SMOTE
from collections import Counter
print(f'Class distribution before SMOTE: {Counter(y)}')
print(f'Class distribution after SMOTE: {Counter(y_resampled)}')

# Step 7: Feature Scaling
# Scale the features using MinMaxScaler to ensure they are on the same scale.
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Step 8: Split Dataset into Training and Test Sets
# Split the dataset into training and testing sets (70% training, 30% testing).
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Output the shapes of the training and test sets
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# Step 9: Train a Decision Tree Model
# Train a basic Decision Tree model on the training set.
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Step 10: Predict Test Set Results
# Make predictions on the test set.
D_pred = dt_model.predict(X_test)
print(np.concatenate((D_pred.reshape(len(D_pred),1), y_test.reshape(len(y_test),1)), 1))

# Step 11: Evaluate the Model
# Confusion matrix and accuracy score for the model.
cm = confusion_matrix(y_test, D_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy_score(y_test, D_pred))

# Step 12: Decision Tree Pruning (Model 1)
# Train a pruned decision tree by limiting depth and split conditions.
dt_model_pruned = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=1)
dt_model_pruned.fit(X_train, y_train)

# Step 13: Evaluate Pruned Tree (Model 1)
D_pred_pruned = dt_model_pruned.predict(X_test)
print("Confusion Matrix for Pruned Tree:\n", confusion_matrix(y_test, D_pred_pruned))
print("Accuracy Score for Pruned Tree:", accuracy_score(y_test, D_pred_pruned))

# Step 14: Visualize the Confusion Matrix
conf_matrix = confusion_matrix(y_test, D_pred)
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
vis.plot(cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Step 15: Feature Importance
# Determine the importance of each feature in the decision tree.
feature_importances = dt_model_pruned.feature_importances_

# Create a DataFrame for feature importance and visualize.
importance_df = pd.DataFrame({'Feature': Data_set.columns[:-1], 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance.
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
plt.title('Feature Importance in Decision Tree Model')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# Step 16: ROC Curve and AUC Score
# Calculate the AUC-ROC for the pruned decision tree.
y_pred_proba = dt_model_pruned.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'DT (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
plt.show()
