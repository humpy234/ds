#Q1
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

plt.hist(df['Pclass'], bins=[0.5, 1.5, 2.5, 3.5], edgecolor='black')
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Passengers in Each Class")
plt.xticks([1, 2, 3])
plt.show()

df['Survived'].value_counts().plot.pie(
    labels=['Not Survived', 'Survived'],
    autopct='%1.1f%%'
)
plt.title("Survival Distribution")
plt.ylabel("")
plt.show()

#csv data
PassengerId,Survived,Pclass,Name,Sex,Age
1,0,3,Mithhu don,male,22
2,1,1,Narendra Modi,female,38
3,1,3,Gyanesh Kumar,female,26


#Q2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("songs.csv")
X = df[['danceability', 'energy', 'tempo']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Polynomial Regression")
plt.show()
