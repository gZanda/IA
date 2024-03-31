import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from memory_profiler import memory_usage

def main_algorithm():

    # Load the CSV
    df = pd.read_csv('iris.csv')

    # Shuffle using a fixed seed
    SEED = 654
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
    k = int(input("Enter the number of neighbors (1,3,5,7): "))
    knn = KNeighborsClassifier( n_neighbors=k )

    # Train
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Add predictions to a new column 
    df_test['Predicted_Species'] = y_pred

    # Results and Precision Metrics
    print(df_test)
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print("Precision:",precision_score(y_test, y_pred, average='weighted'))
    print("Recall:",recall_score(y_test, y_pred, average='weighted'))

    # Confusion Matrix
    true_labels = df_test['Species']
    predicted_labels = df_test['Predicted_Species']
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=df['Species'].unique(), columns=df['Species'].unique())

    print("Confusion Matrix:")
    print(cm_df)

# ------ Performance Metrics ------ #

start_time = time.time()

# Peak memory usage
mem_usage = memory_usage(proc=main_algorithm)
peak_mem = max(mem_usage)
peak_mem_mb = round(peak_mem / 1024, 2)  # Convert from MiB to MB
print("Peak memory usage:", peak_mem_mb, "MB")

# Execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")