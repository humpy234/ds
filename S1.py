#Q1
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = {
    'Product': ['Laptop', 'Mobile', 'Tablet', 'Headphones', 'Smartwatch'],
    'Price': [60000, 25000, 30000, 3000, 8000],
    'Rating': [4.5, 4.3, 4.2, 4.0, 4.1],
    'Sales': [120, 300, 180, 500, 350]
}
df = pd.DataFrame(data)
print(df)
X = df[['Price', 'Rating', 'Sales']]
X = StandardScaler().fit_transform(X)
Z = linkage(X, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=df['Product'].values)
plt.title("Dendrogram of Products")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()

#viva
What is a dendrogram?
A dendrogram is a tree-like diagram used to visualize the grouping of similar products using hierarchical clustering.

Why is StandardScaler() used in this program?
StandardScaler() standardizes the features so that differences in numerical scales, such as price and sales, do not disproportionately affect clustering.


  #Q2 OR
  # Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    s=50
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=250,
    c='red',
    label='Centroids'
)
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.title('K-Means Clustering on Iris Dataset (K=3)')
plt.legend()
plt.show()
print("Cluster Centroids:")
print(centroids)
print("\nNumber of data points in each cluster:")
for i in range(3):
    print(f"Cluster {i}: {(labels == i).sum()} data points")

#viva
1)What is K-Means clustering?
K-Means is an unsupervised machine learning algorithm that divides data into K clusters based on similarity.
  
2)What is a centroid in K-Means clustering?
A centroid is the center point of a cluster, calculated as the mean of all data points assigned to that cluster.


