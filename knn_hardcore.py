import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from memory_profiler import memory_usage

# ------ Functions ------ #

#  Calculate the Euclidean distance 
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

def main_algorithm():

    # 1. Load the CSV data into a DataFrame
    df = pd.read_csv("iris.csv")

    # 2. Shuffle the dataframe usign a fixed seed
    SEED = 654
    np.random.seed(SEED)
    df_shuffled = df.sample(frac=1 , random_state=SEED).reset_index(drop=True)

    # 3. Split into training and testing ( 80/20 )
    df_train = df_shuffled[0:120].copy()
    df_test = df_shuffled[120:].copy()

    # 4. Find the k-nearest neighbors
    k = int(input("Enter the number of neighbors (1,3,5,7): "))

    nearest_neighbors_list = []
    for _, test_row in df_test.iterrows():
        neighbors = k_nearest_neighbors(df_train, test_row, k)
        nearest_neighbors_list.append(neighbors)

    # 5. Predict the TEST species 
    species_predictions = classify_species(df_train, nearest_neighbors_list)
    df_test['Predicted_Species'] = species_predictions

    # 7. Precision Metrics ( Confusion Matrix, Accuracy, Precision, Recall )
    true_labels = df_test['Species']
    predicted_labels = df_test['Predicted_Species']
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())

    # 6. Display prediction results
    with open('results.txt', 'w') as r:
        selected_columns = df_test[['Id', 'Species', 'Predicted_Species']]
        print("Classification: \n",file=r)
        print(selected_columns,file=r)

        print("\nPrecision Metrics: \n",file=r)
        print("Accuracy:", accuracy_score(true_labels, predicted_labels),file=r)
        print("Precision for each class:", precision_score(true_labels, predicted_labels, average='weighted'),file=r)
        print("Recall for each class:", recall_score(true_labels, predicted_labels, average='weighted'),file=r)

        print("\nConfusion Matrix: \n",file=r)
        print(cm_df,file=r)

# ------ Performance Metrics ------ #
    
start_time = time.time()

# Peak memory usage
mem_usage = memory_usage(proc=main_algorithm)
peak_mem = max(mem_usage)
peak_mem_mb = round(peak_mem / 1024, 2)  # Convert from MiB to MB

# Execution time
end_time = time.time()
execution_time = end_time - start_time

# Write to result file
with open('results.txt', 'a') as r:
    print("\nPerformance Metrics: \n",file=r)
    print("Peak memory usage:", peak_mem_mb, "MB",file=r)
    print("Execution time:", execution_time, "seconds",file=r)