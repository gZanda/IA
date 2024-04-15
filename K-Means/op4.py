import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from time import sleep

# Euclidean Distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Input K
k = int(input("Enter the number of clusters: "))

# Load the data
data = pd.read_csv("iris.csv")

# Drop unnecessary columns
data.drop(columns=['Id', 'Species'], inplace=True)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Randomly initialize centroids
centroids = np.random.rand(k, 2) * 10 - 5

# Your data and parameters
max_iters = 100
k = k  # Number of clusters
# Initialize centroids and other necessary variables here

# Create initial plot
plt.figure(figsize=(8, 6))
plt.title('K-means Clustering')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.ion()  # Turn on interactive mode

# Iterate until convergence
for _ in range(max_iters):
    # Assign points to the nearest centroid
    clusters = [[] for _ in range(k)]
    for point in pca_result:
        # Check if centroids list is empty
        if not centroids.any():
            # Handle the case when centroids list is empty
            print("Error: Centroids list is empty.")
            break
        
        # Calculate distances to centroids
        try:
            closest_centroid_idx = min(range(k), key=lambda i: distance(point, centroids[i]))
            clusters[closest_centroid_idx].append(point)
        except IndexError:
            # Handle the case when centroids list index is out of range
            print("Error: Index out of range for centroids list.")
            break
        except Exception as e:
            # Handle any other exceptions
            print(f"An error occurred: {e}")
            break


    # Update centroids
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
    new_centroids = np.array(new_centroids)

    # Check for convergence
    if centroids.shape == new_centroids.shape and np.allclose(centroids, new_centroids):
        break
    
    # Update centroids
    centroids = new_centroids

    # Clear previous plot
    plt.clf()

    # Plot clusters and centroids
    for i in range(k):
        if clusters[i]:
            plt.scatter(*zip(*clusters[i]), label=f'Cluster {i + 1}')

    # Check if centroids array is not empty before plotting
    if centroids.size != 0:
        # Reshape centroids array to maintain 2-dimensional shape
        centroids_reshaped = centroids.reshape(-1, 2)
        plt.scatter(centroids_reshaped[:, 0], centroids_reshaped[:, 1], color='red', marker='x', label='Centroids')

    plt.title('K-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Update plot
    plt.legend()
    plt.draw()
    plt.pause(1)

# Keep the plot window open
plt.ioff()
plt.show()
