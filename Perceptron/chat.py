import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
df_test['Predicted_Species'] = le.inverse_transform(mlp_classifier.predict(X_test))

# Print the DataFrame with predictions
selected_columns = df_test[['Id', 'Species', 'Predicted_Species']]
print(selected_columns.to_string(index=False))

# Evaluate the classifier
accuracy = accuracy_score(y_test, mlp_classifier.predict(X_test))
precision = precision_score(y_test, mlp_classifier.predict(X_test), average='macro')
recall = recall_score(y_test, mlp_classifier.predict(X_test), average='macro')

# Write performance metrics
print(accuracy, precision, recall)
