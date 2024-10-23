# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset
dataset = pd.read_csv('card_transdata.csv')

# Separate the features (X) and the target variable (y)
X = dataset.iloc[:, :-1].values  # All columns except the last as features
y = dataset.iloc[:, -1].values   # The last column as the target variable (fraud or not)

"""## Checking the distribution of the dependent variable"""

# Distribution of target variable 'fraud'
sns.set_style("whitegrid")  # Set the plot style for seaborn

# Plotting the distribution of fraudulent vs non-fraudulent transactions
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
                xytext=(0, 9),
                textcoords='offset points')

plt.show()

"""## Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)"""

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Plot the distribution of the target variable after applying SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Distribution of the Target Variable After SMOTE')
plt.xlabel('FRAUD')
plt.ylabel('Count')
plt.show()

# Output the class distribution before and after applying SMOTE
print(f'Class distribution before SMOTE: {Counter(y)}')
print(f'Class distribution after SMOTE: {Counter(y_resampled)}')

"""## Normalization of feature data"""

# Normalize the feature data using MinMaxScaler to bring all values between 0 and 1
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

"""## Splitting the dataset into the Training set and Test set"""

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Output the shapes of the resulting datasets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

"""## Logistic Regression Model Training"""

from sklearn.linear_model import LogisticRegression

# Train the Logistic Regression model
lg_model = LogisticRegression(random_state=0)
lg_model.fit(X_train, y_train)

"""## Predicting the Test set results"""

# Make predictions on the test data
lg_pred = lg_model.predict(X_test)

# Concatenate the predicted and actual values for comparison
print(np.concatenate((lg_pred.reshape(len(lg_pred),1), y_test.reshape(len(y_test),1)), 1))

"""## Confusion Matrix and Performance Metrics"""

# Generate confusion matrix
cm = confusion_matrix(y_test, lg_pred)
print(cm)

# Output the accuracy score
print("Accuracy:", accuracy_score(y_test, lg_pred))

# Visualize the confusion matrix
confu_matrix = confusion_matrix(y_test, lg_pred)
vis = ConfusionMatrixDisplay(confusion_matrix=confu_matrix, display_labels=[0, 1])
vis.plot(cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Logistic Regression Confusion Matrix')
plt.grid(False)
plt.show()

# Display the classification report, which includes precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, lg_pred))

"""## ROC-AUC Curve and AUC Score"""

from sklearn.metrics import roc_curve, roc_auc_score, auc

# Predict probabilities for the positive class
y_pred_lg_proba = lg_model.predict_proba(X_test)[:, 1]

# Calculate the false positive rate and true positive rate for ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_lg_proba)

# Calculate AUC score
auc_rf = roc_auc_score(y_test, y_pred_lg_proba)
print("AUC RF:", auc_rf)

# Plot the ROC curve
plt.plot(fpr_rf, tpr_rf, label="Lg, auc={:.2f}".format(auc_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc=4)
plt.show()

"""## Cross Validation for Model Validation"""

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation to evaluate the model
scores = cross_val_score(lg_model, X_resampled, y_resampled, cv=5)

# Output the cross-validation scores
print("Cross-Validation Scores:", scores)

# Calculate and print the average cross-validation accuracy
average_accuracy = scores.mean()
print("Average Accuracy:", average_accuracy)
