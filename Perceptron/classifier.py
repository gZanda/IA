import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
iris_data = pd.read_csv('iris.csv')

# Separate features and target
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Convert categorical labels to numerical labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Get predictions
predictions = mlp_classifier.predict(X_test)

# Decode numerical labels back to categorical labels
predicted_species = le.inverse_transform(predictions)

# Add predictions as a new column in the DataFrame
iris_data['Predicted_Species'] = le.inverse_transform(mlp_classifier.predict(X))

# Print the DataFrame with predictions
selected_columns = iris_data[['Id', 'Species', 'Predicted_Species']]
print(selected_columns.to_string(index=False))

# Evaluate the classifier
accuracy = mlp_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)