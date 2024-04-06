import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import os

# ------ Functions ------ #

#  Calculate the Euclidean distance 
def distanciaEuclidiana (linha1, linha2):
    sl = (linha1['SepalLengthCm'] - linha2['SepalLengthCm']) ** 2
    sw = (linha1['SepalWidthCm'] - linha2['SepalWidthCm']) ** 2
    pl = (linha1['PetalLengthCm'] - linha2['PetalLengthCm']) ** 2
    pw = (linha1['PetalWidthCm'] - linha2['PetalWidthCm']) **2
    return math.sqrt(sl + sw + pl + pw)

# find the K-nearest neighbors and make a list of they indexes
def k_nearest_neighbors(train_df, test_row, k):
    distances = []
    for idx, train_row in train_df.iterrows():
        dist = distanciaEuclidiana(train_row, test_row)
        distances.append((idx, dist))  # Store index and distance
    distances.sort(key=lambda x: x[1])  # Sort by distance
    neighbors = [idx for idx, _ in distances[:k]]  # Store only index
    return neighbors

# Classify the species based on the most common species in the neighbors
def classify_species(train_df, nearest_neighbors_list):
    species_list = []
    for neighbors in nearest_neighbors_list:
        species_counts = train_df.loc[neighbors, 'Species'].value_counts()
        most_common_species = species_counts.idxmax()
        species_list.append(most_common_species)
    return species_list

# ------ Main Code ------ #

def main_algorithm(k_input):

    # Load the CSV data into a DataFrame
    df = pd.read_csv("iris.csv")

    # Shuffle the dataframe usign a fixed seed
    SEED = 123
    np.random.seed(SEED)
    df_shuffled = df.sample(frac=1 , random_state=SEED).reset_index(drop=True)

    # Split into training and testing ( 80/20 )
    df_train = df_shuffled[0:120].copy()
    df_test = df_shuffled[120:].copy()

    # Find the k-nearest neighbors
    k = k_input

    nearest_neighbors_list = []
    for _, test_row in df_test.iterrows():
        neighbors = k_nearest_neighbors(df_train, test_row, k)
        nearest_neighbors_list.append(neighbors)

    # Predict the TEST species 
    species_predictions = classify_species(df_train, nearest_neighbors_list)
    df_test['Predicted_Species'] = species_predictions

    # Confusion Matrix
    true_labels = df_test['Species']
    predicted_labels = df_test['Predicted_Species']
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())

    # Write results and Precision Metrics
    with open('results.txt', 'a') as r:
        print("\n\n########################################################################################################################\n\n",file=r)
        print(f"** HARDCORE KNN ALGORITHM ( K={k}) ** \n",file=r)

        selected_columns = df_test[['Id', 'Species', 'Predicted_Species']]
        print("# ------ Classification ------# \n",file=r)
        print(selected_columns.to_string(index=False),file=r)

        print("\n# ------ Precision Metrics ------# \n",file=r)
        print("Accuracy:", accuracy_score(true_labels, predicted_labels),file=r)
        print("Precision:", precision_score(true_labels, predicted_labels, average='weighted'),file=r)
        print("Recall:", recall_score(true_labels, predicted_labels, average='weighted'),file=r)

        print("\n# ------ Confusion Matrix ------# \n",file=r)
        print(cm_df,file=r)

# ------ Main Algorithm Call ------ #

# Delete the "results.txt" file 
if os.path.exists('results.txt'):
    os.remove('results.txt')        

# K's
list_of_k = [1, 3, 5, 7]

# Message to User
print("Running the Hardcore KNN Algorithm...")
    
# Start measure execution time
start_time = time.time()

# Main Code Call
for k in list_of_k:
    main_algorithm(k)

# End measure execution time
end_time = time.time()
execution_time = end_time - start_time

# Write performance metrics
with open('results.txt', 'a') as r:
    print("\n\n########################################################################################################################\n",file=r)
    print("\n# ------ Performance Metrics KNN HARDCORE ------# \n",file=r)
    print("Execution time:", execution_time, "seconds",file=r)