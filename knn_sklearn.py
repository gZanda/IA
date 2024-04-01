import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import os

def main_algorithm(k_input):

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
    k = k_input
    knn = KNeighborsClassifier( n_neighbors=k )

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

    # Write results and Precision Metrics
    with open('results.txt', 'a') as r:
        print("\n\n########################################################################################################################\n\n",file=r)
        print(f"** SKLEARN KNN ALGORITHM ( K={k}) ** \n",file=r)

        selected_columns = df_test[['Id', 'Species', 'Predicted_Species']]
        print("# ------ Classification ------# \n",file=r)
        print(selected_columns.to_string(index=False),file=r)

        print("\n# ------ Precision Metrics ------# \n",file=r)
        print("Accuracy:", accuracy_score(y_test, y_pred),file=r)
        print("Precision:", precision_score(y_test, y_pred, average='weighted'),file=r)
        print("Recall:", recall_score(y_test, y_pred, average='weighted'),file=r)

        print("\n# ------ Confusion Matrix ------# \n",file=r)
        print(cm_df,file=r)

# ------ Main Algorithm Call ------ #
             
# K's
list_of_k = [1, 3, 5, 7]
    
# Start measure execution time
start_time = time.time()

# Message to User
print("Running the SKLEARN KNN Algorithm...")

# Main Code Call
for k in list_of_k:
    main_algorithm(k)

# End measure execution time
end_time = time.time()
execution_time = end_time - start_time

# Write performance metrics
with open('results.txt', 'a') as r:
    print("\n# ------ Performance Metrics ------# \n",file=r)
    print("Execution time:", execution_time, "seconds",file=r)