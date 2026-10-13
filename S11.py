#Q1
import matplotlib.pyplot as plt

departments = ['HR', 'Finance', 'IT', 'Sales', 'Marketing']
employees = [20, 15, 40, 25, 30]

plt.bar(departments, employees, color='skyblue')
plt.xlabel("Departments")
plt.ylabel("Number of Employees")
plt.title("Employees in Different Departments")
plt.show()



#Q2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("smartphone.csv")
X = df.drop("price_range", axis=1)
y = df["price_range"]
X = pd.get_dummies(X)
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
print("F1-Score:", f1_score(y_test, y_pred, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

