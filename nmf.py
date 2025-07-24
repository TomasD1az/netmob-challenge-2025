from sklearn.decomposition import NMF
from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate a synthetic non-negative dataset

# Initialize NMF
n_components = 3  # number of clusters/components
model = NMF(n_components=n_components, init='random', random_state=42)

# Apply NMF
W = model.fit_transform(X)  # shape: (samples, components)
H = model.components_       # shape: (components, features)

# Cluster assignment: each sample is assigned to the component with the highest weight
clusters = np.argmax(W, axis=1)

# Show clustering result
print("Cluster assignments:\n", clusters)

# Optional: visualize the clusters in 2D (if using dimensionality reduction)
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2).fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
plt.title("NMF-based Clustering Visualization")
plt.show()
