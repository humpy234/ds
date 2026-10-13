#Q1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("weather.csv")

df = df[['Temperature', 'Humidity', 'Rainfall', 'Wind Speed']]

corr = df.corr()

print("Correlation Matrix:")
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Weather Parameters Correlation Heatmap")
plt.show()

#csv file
Temperature,Humidity,Rainfall,Wind Speed
25,60,10,12
28,65,15,14
30,70,20,16

#Q2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    'Student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Attendance': [90, 85, 80, 60, 55, 50, 40, 45],
    'Marks': [95, 88, 85, 70, 65, 60, 45, 50]
})

X = df[['Attendance', 'Marks']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Student Dataset with Clusters:")
print(df)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(df['Attendance'], df['Marks'], c=df['Cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red',
            marker='X', s=200, label='Centroids')

plt.xlabel('Attendance (%)')
plt.ylabel('Marks')
plt.title('K-Means Clustering of Students (K=3)')
plt.legend()
plt.show()
