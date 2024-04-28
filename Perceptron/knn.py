import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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

# Feeatures / Target for TRAIN
X_train = df_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_train = df_train['Species']

# Features / Target for TEST
X_test = df_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_test = df_test['Species']

# Initialize with k 
knn = KNeighborsClassifier(5)

# Train
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Add predictions to a new column 
df_test['Predicted_Species'] = y_pred

# Confusion Matrix
true_labels = df_test['Species']
predicted_labels = df_test['Predicted_Species']
cm = confusion_matrix(true_labels, predicted_labels)
cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())

# Print the DataFrame with predictions
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
print("/n")