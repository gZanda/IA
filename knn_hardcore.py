import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("iris.csv")

# ------ Functions ------ #

# Euclidian Distance
def distanciaEuclidiana (linha1, linha2):
    sl = (linha1['SepalLengthCm'] - linha2['SepalLengthCm']) ** 2
    sw = (linha1['SepalWidthCm'] - linha2['SepalWidthCm']) ** 2
    pl = (linha1['PetalLengthCm'] - linha2['PetalLengthCm']) ** 2
    pw = (linha1['PetalWidthCm'] - linha2['PetalWidthCm']) **2
    return math.sqrt(sl + sw + pl + pw)

# find the K-nearest neighbors 
def k_nearest_neighbors(train_df, test_row, k):
    distances = []
    for idx, train_row in train_df.iterrows():
        dist = distanciaEuclidiana(train_row, test_row)
        distances.append((idx, dist))  # Store index and distance
    distances.sort(key=lambda x: x[1])  # Sort by distance
    neighbors = [idx for idx, _ in distances[:k]]  # Store only index
    return neighbors

# Classification of the species
def classify_species(train_df, nearest_neighbors_list):
    species_list = []
    for neighbors in nearest_neighbors_list:
        species_counts = train_df.loc[neighbors, 'Species'].value_counts()
        most_common_species = species_counts.idxmax()
        species_list.append(most_common_species)
    return species_list

# ------ Main Code ------ #

# 1. Shuffle the dataframe usign a fixed seed
SEED = 654
np.random.seed(SEED)
df_shuffled = df.sample(frac=1 , random_state=SEED).reset_index(drop=True)

# 2. Split the dataframe into training and testing data ( 80/20 )
df_train = df_shuffled[0:120].copy()
df_test = df_shuffled[120:].copy()

# 3. Find the k-nearest neighbors
k = int(input("Enter the number of neighbors (1,3,5,7): "))

nearest_neighbors_list = []
for _, test_row in df_test.iterrows():
    neighbors = k_nearest_neighbors(df_train, test_row, k)
    nearest_neighbors_list.append(neighbors)

# 4. Classify the species 
species_predictions = classify_species(df_train, nearest_neighbors_list)
df_test['Predicted_Species'] = species_predictions

# 5. Display classification results
print(df_test)

# 6. Confusion Matrix
true_labels = df_test['Species']
predicted_labels = df_test['Predicted_Species']
cm = confusion_matrix(true_labels, predicted_labels)
cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())

# 7. Other Metrics

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Calculate precision for each class
overall_precision = precision_score(true_labels, predicted_labels, average='weighted')
print("Precision for each class:", overall_precision)

# Calculate recall for each class
overall_recall = recall_score(true_labels, predicted_labels, average='weighted')
print("Recall for each class:", overall_recall)

print("Confusion Matrix:")
print(cm_df)
