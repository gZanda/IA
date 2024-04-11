import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# Create Initial plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df['Component 1'], pca_df['Component 2'], alpha=1, color='blue')
plt.title('Two dimensional Iris PCA')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# Display the plot
plt.draw()
plt.pause(3)

# Clear the plot
plt.clf()

# START K-MEANS CLUSTERING

