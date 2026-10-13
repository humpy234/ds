#Q1
import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Car_Model': ['Car A', 'Car B', 'Car C', 'Car D', 'Car E'],
    'Engine_Size': [1.0, 1.5, 2.0, 2.5, 3.0],
    'Fuel_Efficiency': [22, 20, 17, 15, 12]
}
df = pd.DataFrame(data)
print(df)
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.scatter(df['Engine_Size'], df['Fuel_Efficiency'], c=colors)
plt.title('Engine Size vs Fuel Efficiency')
plt.xlabel('Engine Size (litres)')
plt.ylabel('Fuel Efficiency (km/L)')
plt.show()

#viva
What is a scatter plot?
A scatter plot is a graph that uses points to show the relationship between two numerical variables.

What is the purpose of different colors in a scatter plot?
Different colors help distinguish individual data points and make the graph easier to understand.


#Q2
  import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv('customer.csv')
df = df.dropna()
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop('Segment', axis=1)
y = df['Segment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#csv
Customer_ID,Age,Annual_Income,Spending_Score,Segment
1,20,20000,85,High
2,25,25000,80,High

#viva
What is KNN?
KNN (K-Nearest Neighbors) is a supervised machine learning algorithm used for classification and regression.

How is the accuracy of a KNN model evaluated?
Accuracy measures the proportion of correctly predicted results out of the total predictions. A confusion matrix shows correct and incorrect classifications
