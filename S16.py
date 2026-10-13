#Q1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Study Hours': [2, 4, 6, 8, 5],
    'Attendance': [60, 70, 80, 90, 75],
    'Assignment Score': [50, 60, 75, 90, 70],
    'Exam Score': [45, 55, 70, 95, 65]
})

corr = df.corr()

print("Correlation Matrix:")
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Student Performance Correlation Heatmap")
plt.show()

#Q2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

df = pd.DataFrame({
    'Temperature': [25, 30, 35, 40, 45],
    'Humidity': [40, 50, 60, 70, 80],
    'WindSpeed': [5, 10, 15, 20, 25],
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

#viva
Q: What is Min-Max Scaling?
Ans: It converts values into a range between 0 and 1.

Q: Which function is used for standardization?
Ans: StandardScaler().

Q: Which function is used for normalization?
Ans: Normalizer().



