import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_sklearn_algorithm(k):

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

    # Apply K-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=None, n_init=10)
    kmeans.fit(pca_result)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(pca_result, kmeans.labels_)
    print(f"K-Means Sklearn silhouette score = {silhouette_avg}")

    # Plot the clusters
    plt.figure(figsize=(5, 4))
    plt.title('K-means Sklearn')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Iterate through each cluster label and plot the corresponding data points separately
    for i in range(k):
        plt.scatter(pca_result[kmeans.labels_ == i, 0], pca_result[kmeans.labels_ == i, 1], label=f'Cluster {i + 1}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', label='Centroids', s=100)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    k = int(sys.argv[1])
    run_sklearn_algorithm(k)