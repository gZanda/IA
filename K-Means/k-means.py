import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import StandardScaler
import numpy as np

# Euclidian Distance betwwen two points
def distance(point1, point2):
    return np.sqrt((point1['X'] - point2['X'])**2 + (point1['Y'] - point2['Y'])**2)

# Input K
k = int(input("Enter the number of clusters: "))

# Load the data
data = pd.read_csv("iris.csv")

# Drop the 'Id' and 'Species' columns
data.drop(columns=['Id', 'Species'], inplace=True)

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame
pca_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])

# Find the Biggest and Smallest values of the dataframe
max_range = pca_df.max().max()
min_range = pca_df.min().min()

# Set K random centroids in range of max + 2 and min - 2
centroids = pd.DataFrame(columns=['X', 'Y'])
for i in range(k):
    x = random.uniform(min_range - 1, max_range + 1)
    y = random.uniform(min_range - 1, max_range + 1)
    centroids = centroids._append({'X': x, 'Y': y}, ignore_index=True)

# Print the centroids
print(centroids)

# Create plot of iris pca data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df['Component 1'], pca_df['Component 2'], alpha=1, color='blue')
plt.title('Two dimensional Iris PCA')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.xlim(min_range - 2, max_range + 2)
plt.ylim(min_range - 2, max_range + 2)

# Display the initial plot
plt.draw()
plt.pause(3)

# Add the centroids 
for i in range(k):
    plt.scatter(centroids['X'][i], centroids['Y'][i], color='red', marker='x', label='Centroid ' + str(i + 1))
    plt.draw()
    plt.pause(1)

# plt.scatter(2, 2, color='red', marker='x', label='Specific Point')

# Print df
print(pca_df.head())

# Display the plot
plt.draw()
plt.pause(3)

# Clear the plot
# plt.clf()

# START K-MEANS CLUSTERING

# Display the plot until the user closes it
plt.show()
