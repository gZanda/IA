import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Print the header
print("###--- MLP ---###\n")

# Load the CSV
df = pd.read_csv('iris.csv')

# Shuffle using a fixed seed
SEED = 123
np.random.seed(SEED)
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split (80/20)
df_train = df_shuffled.iloc[:120].copy()
df_test = df_shuffled.iloc[120:].copy()

# Features / Target for TRAIN
X_train = df_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_train = df_train['Species']

# Features / Target for TEST
X_test = df_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_test = df_test['Species']

# Convert categorical labels to numerical labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Initialize the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Get predictions and place them in a new column
y_pred = mlp_classifier.predict(X_test)

# Print the DataFrame with predictions
df_test['Predicted_Species'] = le.inverse_transform(y_pred)
selected_columns = df_test[['Id', 'Species', 'Predicted_Species']]
print(selected_columns.to_string(index=False))

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Line
print("\n--------------------------------------------------")

# Write performance metrics
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\n--------------------------------------------------")

# Confusion Matrix
true_labels = df_test['Species']
predicted_labels = df_test['Predicted_Species']
cm = confusion_matrix(true_labels, predicted_labels)
cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())
print("\nConfusion Matrix:")
print(cm_df)
print("\n")