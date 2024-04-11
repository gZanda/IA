import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('iris.csv')

# Extract features (excluding 'Id' and 'Species' columns)
features = df.drop(['Id', 'Species'], axis=1)

# Perform PCA to reduce dimensionality to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print(pca_df)

# Plot the points using Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.grid(True)
plt.show()
