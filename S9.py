#Q1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = {
    'Product_ID': [1, 2, 3, 4, 5],
    'Price': [100, 200, 300, 400, 500],
    'Quantity_Sold': [10, 20, 30, 40, 50],
    'Discount': [5, 10, 15, 20, 25],
    'Revenue': [950, 3600, 7650, 12800, 18750]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("\nTransformed DataFrame:")
print(scaled_df)

#viva
What is Min-Max Scaling?
Min-Max Scaling is a data transformation technique that scales numerical values to a fixed range, usually between 0 and 1.

Which Python function is used for Min-Max Scaling?
The MinMaxScaler() function from sklearn.preprocessing is used to perform Min-Max Scaling.

#Q2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
df = pd.read_csv('smartphones.csv')
X = df[['RAM', 'Storage', 'Battery', 'Camera']]
y = df['Price_Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#csv
RAM,Storage,Battery,Camera,Price_Category
2,32,3000,8,Low
3,64,3500,12,Low

#viva
What is Logistic Regression?
Logistic Regression is a supervised machine learning algorithm used to classify data into different categories.

Why is F1-Score used to evaluate the model?
F1-Score combines precision and recall to measure the model's classification performance.
  
