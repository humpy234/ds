#Q!
import matplotlib.pyplot as plt

courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]

colors = ["red", "blue", "green", "orange", "purple"]

plt.bar(courses, students, color=colors)
plt.title("Students Enrolled in Different Courses")
plt.xlabel("Courses")
plt.ylabel("Number of Students")
plt.show()

#Q2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df1 = pd.DataFrame({
    'Brand': ['Samsung', 'Apple', 'Vivo', 'Redmi'],
    'Model': ['A15', 'iPhone15', 'V29', 'Note13'],
    'Price': [15000, 70000, 30000, 18000],
    'RAM': [6, 8, 8, 6]
})

df2 = pd.DataFrame({
    'Model': ['A15', 'iPhone15', 'V29', 'Note13'],
    'Storage': [128, 256, 256, 128],
    'Battery_Capacity': [5000, 3349, 4600, 5000],
    'Operating_System': ['Android', 'iOS', 'Android', 'Android']
})
print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)
df = pd.merge(df1, df2, on='Model')
print("\nMerged DataFrame:")
print(df)
le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])
df = pd.get_dummies(df, columns=['Operating_System'], dtype=int)
print("\nTransformed DataFrame:")
print(df)
