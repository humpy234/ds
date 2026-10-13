#Q1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.DataFrame({
    'Product_name': ['Mobile', 'Laptop', 'Tablet', 'TV', 'Headphones'],
    'Price': [15000, 60000, 25000, 40000, 2000],
    'Rating': [4.2, 4.5, 4.0, 4.3, 4.1],
    'Number_of_Sales': [500, 200, 350, 150, 800]
})

print("Product Data:")
print(df)

X = df[['Price', 'Rating', 'Number_of_Sales']]
X = StandardScaler().fit_transform(X)

Z = linkage(X, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=df['Product_name'].values)
plt.title("Dendrogram of Similar Products")
plt.xlabel("Product Name")
plt.ylabel("Euclidean Distance")
plt.show()

#Q2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

df = pd.DataFrame({
    'Temperature': [25, 30, 35, 40, 45],
    'Humidity': [40, 50, 60, 70, 80],
    'Wind Speed': [5, 10, 15, 20, 25],
    'Pressure': [1000, 1005, 1010, 1015, 1020],
    'Rainfall': [10, 20, 30, 40, 50]
})

print("Original Data:")
print(df)

cols = df.columns

print("\nMin-Max Scaling:")
print(pd.DataFrame(MinMaxScaler().fit_transform(df), columns=cols).round(2))

print("\nStandardization:")
print(pd.DataFrame(StandardScaler().fit_transform(df), columns=cols).round(2))

print("\nNormalization:")
print(pd.DataFrame(Normalizer().fit_transform(df), columns=cols).round(2))
