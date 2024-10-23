# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance

# Load the dataset
# Assume 'card_transdata.csv' contains transaction data with the last column being the target variable (fraud or non-fraud)
dataset = pd.read_csv('card_transdata.csv')

# Extracting features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # All columns except the last one are features
y = dataset.iloc[:, -1].values   # The last column is the target variable (fraud indicator)

"""## Distribution Of The Dependent Variable"""

# Visualizing the distribution of fraudulent vs. non-fraudulent transactions
# Setting the style for seaborn plots
sns.set_style("whitegrid")

# Create a bar plot for the 'fraud' column
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=dataset, x='fraud')
plt.title('Non-Fraudulent Transactions vs. Fraudulent')
plt.xlabel('FRAUD')
plt.ylabel('Count')

# Adding data labels to the plot
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

# Show the plot
plt.show()

"""## Normalization"""

# Normalizing the feature variables using MinMaxScaler to scale values between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Display the normalized features
print(X)

"""## Splitting the dataset into the Training set and Test set"""

# Split the dataset into training and test sets
# 70% training data, 30% test data, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the training and test sets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

"""## Training KNN Model"""

# Initializing the K-Nearest Neighbors classifier with 5 neighbors and Minkowski distance
KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Fit the KNN model on the training data
KNN_classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

# Predicting the target variable (fraud or non-fraud) for the test set
y_pred_K = KNN_classifier.predict(X_test)

# Concatenating predicted and actual values for comparison
print(np.concatenate((y_pred_K.reshape(len(y_pred_K), 1), y_test.reshape(len(y_test), 1)), 1))

"""## Making the Confusion Matrix"""

# Create a confusion matrix to evaluate model performance
cm = confusion_matrix(y_test, y_pred_K)
print(cm)

# Print the accuracy of the model
print(f"Accuracy: {accuracy_score(y_test, y_pred_K):.2f}")

"""## Algorithm To Find The Best K - Number"""

# Initialize a list to store error rates for different values of K
error_rate = []

# Loop over a range of K values to find the best one (1 to 40)
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))  # Calculate and store error rate

# Plotting the error rate for different values of K
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

"""## Confusion Matrix"""

# Visualize the confusion matrix using Seaborn and Matplotlib
conf_matrix_K = confusion_matrix(y_test, y_pred_K)
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_K, display_labels=[0, 1])
vis.plot(cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('K Nearest Neighbor Confusion Matrix')
plt.grid(False)
plt.show()

# Display a classification report for precision, recall, and f1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred_K))

"""## AUC-ROC"""

# Calculate and plot the ROC-AUC curve
y_pred_kn_proba = KNN_classifier.predict_proba(X_test)[:, 1]  # Get the probability predictions for the positive class
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_K)
auc_rf = roc_auc_score(y_test, y_pred_K)
print(f"AUC RF: {auc_rf:.2f}")

# Plot the ROC curve
plt.plot(fpr_rf, tpr_rf, label=f"KNN, AUC={auc_rf:.2f}")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K Nearest Neighbor ROC Curve')
plt.legend(loc=4)
plt.show()

"""## Feature Importance"""

# Compute feature importance using permutation importance method
result = permutation_importance(KNN_classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

# Create a DataFrame to visualize feature importance
importance_df = pd.DataFrame({'Feature': dataset.columns[:-1], 'Importance': result.importances_mean})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance in KNN Model')
plt.gca().invert_yaxis()  # Invert the y-axis for better readability
plt.show()
