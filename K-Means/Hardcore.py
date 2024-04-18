import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_custom_kmeans_algorithm(k):

    # Euclidean Distance between two points
    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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

    # Tolerance for centroid updates and iterations
    max_iters = 100
    tolerance = 1e-4  

    # Set seed for reproducibility
    np.random.seed(60)

    # Initialize centroids list with random values ( from -5 to 5 )
    centroids = np.random.rand(k, 2) * 10 - 5

    # Create initial base plot 
    plt.figure(figsize=(5, 4))
    plt.title('K-means Hardcore')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.ion()  # Turn on interactive mode

    # Insert initial data points  
    plt.scatter(pca_result[:, 0], pca_result[:, 1], color='grey', label='Initial Data Points')
    plt.legend()
    plt.draw()
    plt.pause(5)

    # Insert the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Initial Centroids', s=100 )
    plt.legend()
    plt.draw()
    plt.pause(3)

    # Iterate until convergence
    for _ in range(max_iters):

        # Assign points to the nearest centroid
        clusters = [[] for _ in range(k)]
        for point in pca_result:
            
            # Calculate distances to centroids
            closest_centroid_idx = min(range(k), key=lambda i: distance(point, centroids[i]))
            clusters[closest_centroid_idx].append(point)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        # Check for convergence ( stop condition )
        if centroids.shape == new_centroids.shape and np.linalg.norm(centroids - new_centroids) < tolerance:
            break

        # Pad new_centroids array with zeros to match the number of centroids
        if new_centroids.shape[0] < k:
            padding = np.zeros((k - new_centroids.shape[0], 2))
            new_centroids = np.vstack([new_centroids, padding])

        # Update centroids
        centroids = new_centroids

        # Clear previous points on graph
        plt.clf()

        # Plot clusters and centroids
        for i in range(k):
            if clusters[i]:
                plt.scatter(*zip(*clusters[i]), label=f'Cluster {i + 1}')
        if centroids.size != 0:
            # Reshape centroids array to maintain 2-dimensional shape
            centroids_reshaped = centroids.reshape(-1, centroids.shape[1])
            plt.scatter(centroids_reshaped[:, 0], centroids_reshaped[:, 1], color='black', marker='x', label='Centroids', s=100 )

        # Plot again with correct data
        plt.title('K-means Hardcore')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.draw()
        plt.pause(2)

    # Calculate cluster labels
    cluster_labels = np.zeros(len(pca_result))
    for i, cluster in enumerate(clusters):
        for point in cluster:
            point_idx = np.where((pca_result == point).all(axis=1))[0][0]
            cluster_labels[point_idx] = i

    # Calculate silhouette score
    silhouette_avg = silhouette_score(pca_result, cluster_labels)
    print(f"K-Means Hardcore silhouette score = {silhouette_avg}")

    plt.ioff()  # Turn off interactive mode at the end
    plt.show()

if __name__ == "__main__":
    k = int(sys.argv[1])
    run_custom_kmeans_algorithm(k)
