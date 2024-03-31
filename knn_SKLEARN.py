import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Data reading
dados = pd.read_csv('iris.csv')

# Separating features and target
X = dados.drop('Species', axis=1)
y = dados['Species']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)

# Instantiating the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Training the model
knn.fit(X_train, y_train)

# Making predictions
y_pred = knn.predict(X_test)

# Assessing model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo KNN:", accuracy)
print("DATA FRAME X :", X)
print("DATA FRAME X TESTE :", X_test)
print("DATA FRAME X TREINO :", X_train)