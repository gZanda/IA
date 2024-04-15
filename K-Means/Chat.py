import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_clusters(data, centroids, labels, ax):
    ax.clear()
    ax.set_title('K-means Clustering')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids', s=100)
    ax.legend()
    plt.pause(2)

k = int(input("Enter the number of clusters: "))

data = pd.read_csv("iris.csv")

data.drop(columns=['Id', 'Species'], inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pca_result)

fig, ax = plt.subplots(figsize=(8, 6))
plot_clusters(pca_result, kmeans.cluster_centers_, kmeans.labels_, ax)

for _ in range(5):  # Change the number of iterations as needed
    kmeans.fit(pca_result)
    plot_clusters(pca_result, kmeans.cluster_centers_, kmeans.labels_, ax)

plt.show()
