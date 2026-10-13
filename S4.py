#Q1
import matplotlib.pyplot as plt
payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
customers = [150, 80, 60, 40, 30]

plt.pie(customers, labels=payment_methods, autopct='%1.1f%%', startangle=90)
#viva
1)What is a Pie Chart?
A pie chart is a circular graph used to represent data as proportions or percentages of a whole.

2)What is the purpose of autopct='%1.1f%%' in Matplotlib?
It displays the percentage of each category on the pie chart up to one decimal place.

  #Q2 OR
  import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('Ecommerce_Customers.csv')
print("Dataset:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
features = ['Age', 'Annual Income', 'Spending Score']
X = df[features].copy()
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nCustomer Clusters:")
print(df[['Age', 'Annual Income', 'Spending Score', 'Cluster']])
print("\nCluster Characteristics:")
print(df.groupby('Cluster')[features].mean().round(2))
plt.scatter(X['Annual Income'], X['Spending Score'],
            c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('E-commerce Customer Segmentation')
plt.show()

#viva
What is K-Means Clustering?
K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.
  
Why is StandardScaler() used in K-Means?
StandardScaler() scales numerical features to a similar range so that features with larger values do not dominate the clustering process.
  
