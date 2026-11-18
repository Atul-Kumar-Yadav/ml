# Block 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Block 2
from google.colab import drive
import os

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Define the file path (verify if there should be a space after 'ML EXP')
file_path = "/content/drive/MyDrive/ML EXP /iris.csv"

# 3. Check if file exists and load it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
else:
    print(f"Error: File not found at '{file_path}'.")
    print("Please check if the folder name contains spaces, or if the file name is correct.")

# Block 3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# Fallback: if 'df' was not successfully loaded from Drive, load it from sklearn
try:
    df
except NameError:
    print("Warning: 'df' is not defined because the file was not found in Drive. Loading built-in Iris dataset as fallback.")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[target] for target in iris.target]

# Separate features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

print("Data split into training and testing sets successfully.")


# Block 4
# Initialize and train the RandomForest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

print("RandomForest Classifier trained successfully.")


# Block 5
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# Optional: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Block 6

# Get feature importances from the trained RandomForest Classifier
feature_importances = rf_classifier.feature_importances_

# Get feature names from the DataFrame (X)
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances in RandomForest Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# Block 7
from sklearn.tree import plot_tree

# Select a few trees to visualize (e.g., the first 3 trees)
num_trees_to_plot = 3

plt.figure(figsize=(25, 20))
for i, tree in enumerate(rf_classifier.estimators_[:num_trees_to_plot]):
    plt.subplot(1, num_trees_to_plot, i + 1)
    plot_tree(tree, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree {i+1}')
plt.tight_layout()
plt.show()








