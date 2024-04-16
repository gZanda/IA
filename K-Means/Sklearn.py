import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_sklearn_algorithm(k):
    data = pd.read_csv("iris.csv")
    data.drop(columns=['Id', 'Species'], inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=k, random_state=None, n_init=10)
    kmeans.fit(pca_result)

    plt.figure(figsize=(8, 6))
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
