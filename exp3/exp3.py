# Block 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/drive/MyDrive/ML EXP /iris.csv")
df.head()

# Block 2
display(df['species'].value_counts())

# Block 3
# Separate features (X) and target (y)*
X = df.drop('species', axis=1)
y = df['species']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

display(X.head())
display(y.head())


# Block 4
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Original species labels:", le.classes_)
print("Encoded target (y_encoded) head:", y_encoded[:5])
print("Encoded target (y_encoded) unique values:", np.unique(y_encoded))

# Block 5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Block 6
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier
# Using random_state for reproducibility
cart_model = DecisionTreeClassifier(random_state=42)

# Train the model
cart_model.fit(X_train, y_train)

print("CART model trained successfully!")

# Block 7
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = cart_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Block 8
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 15))
plot_tree(cart_model,
          feature_names=X.columns,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Iris Dataset')
plt.show()




