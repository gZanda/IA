import numpy as np
import pandas as pd
import math

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
k = 2
nearest_neighbors_list = []
for _, test_row in df_test.iterrows():
    neighbors = k_nearest_neighbors(df_train, test_row, k)
    nearest_neighbors_list.append(neighbors)

# 4. Classify the species 
species_predictions = classify_species(df_train, nearest_neighbors_list)
df_test['Predicted_Species'] = species_predictions

# 5. Display classification results
print(df_test)
print("Accuracy:", (df_test['Species'] == df_test['Predicted_Species']).mean())