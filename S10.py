#Q1
import matplotlib.pyplot as plt
transport = ["Train", "Bicycle", "Bike", "Car", "Bus"]
employees = [220, 60, 70, 80, 40]
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.pie(employees, labels=transport, colors=colors,
        autopct='%1.1f%%', startangle=90)
plt.title("Preferred Mode of Transportation")
plt.show()

#Viva
What is a Pie Chart?
A pie chart is a circular graph used to represent data as percentages or proportions of a whole.

Which function is used to display percentages in a Pie Chart?
The autopct='%1.1f%%' parameter in plt.pie() displays the percentage of each category up to one decimal place.

#Q2
  import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('housing.csv')
df = df.dropna()
X = df.drop('Price', axis=1)
y = df['Price']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('housing.csv')

df = df.dropna()

X = df.drop('Price', axis=1)
y = df['Price']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2 Score:", r2_score(y_test, y_pred))

#csv
Area,Bedrooms,Bathrooms,Age,Location,Price
1000,2,1,5,Urban,2500000
1200,2,2,4,Urban,3000000

#viva
What is Random Forest Regression?
Random Forest Regression is a machine learning algorithm that uses multiple decision trees to predict continuous values, such as house prices.

What is the R² Score?
R² Score measures how well the model predicts house prices. A score closer to 1 indicates better model performance.
  
print("R2 Score:", r2_score(y_test, y_pred))
