#Q1
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Fuel_Efficiency': [15, 18, 20, 22, 80],
    'Engine_Power': [100, 120, 150, 180, 500],
    'Vehicle_Weight': [1000, 1200, 1400, 1600, 1800],
    'Engine_Size': [1000, 1200, 1500, 1800, 2000],
    'Acceleration': [12, 10, 9, 8, 7]
})

print("Vehicle Data:")
print(df)
plt.boxplot(df['Fuel_Efficiency'])
plt.title("Fuel Efficiency Box Plot")
plt.ylabel("Fuel Efficiency")
plt.show()
plt.boxplot(df['Engine_Power'])
plt.title("Engine Power Box Plot")
plt.ylabel("Engine Power")
plt.show()
for col in ['Fuel_Efficiency', 'Engine_Power']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print("\nOutliers in", col, ":")
    print(outliers[col])

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
print("\nMin-Max Scaling:")
print(pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns))
print("\nStandardization:")
print(pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns))
print("\nNormalization:")
print(pd.DataFrame(Normalizer().fit_transform(df), columns=df.columns))
