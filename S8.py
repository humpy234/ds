#Q1
import pandas as pd
import folium

data = {
    'Latitude': [35.7, 36.1, 34.5, 37.2, 33.8],
    'Longitude': [139.7, 140.2, 138.5, 141.0, 135.5],
    'Depth': [10, 20, 15, 30, 25],
    'Magnitude': [4.2, 5.5, 3.8, 6.5, 7.2]
}

df = pd.DataFrame(data)
print(df)

m = folium.Map(location=[36, 139], zoom_start=5)

for i in range(len(df)):
    color = 'green' if df['Magnitude'][i] < 4 else 'orange' if df['Magnitude'][i] < 6 else 'red'

    folium.CircleMarker(
        location=[df['Latitude'][i], df['Longitude'][i]],
        radius=df['Magnitude'][i] * 2,
        popup=f"Magnitude: {df['Magnitude'][i]}, Depth: {df['Depth'][i]} km",
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)

m.save('earthquake_map.html')

print("Earthquake map saved as earthquake_map.html")

#viva
What is Geospatial Visualization?
Geospatial visualization represents geographical data on a map using locations such as latitude and longitude.

Why is Folium used in this program?
Folium is a Python library used to create interactive maps and display earthquake locations with different colors based on their magnitude.


  #Q2
  import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv('Salary.csv')
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted Salaries:")
print(y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Prediction Using Linear Regression')
plt.legend()
plt.show()

#csv
YearsExperience,Salary
1.1,39343
1.3,46205

#viva
What is Simple Linear Regression?
Simple Linear Regression is a machine learning algorithm used to predict a dependent variable using one independent variable.

What is Mean Squared Error (MSE)?
MSE measures the average squared difference between actual and predicted values. A lower MSE indicates better prediction accuracy.
Explanation
