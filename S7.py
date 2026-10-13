#Q1
import matplotlib.pyplot as plt
courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.bar(courses, students, color=colors)
plt.title("Students Enrolled in Different Courses")
plt.xlabel("Courses")
plt.ylabel("Number of Students")
plt.show()

#viva
What is a Bar Chart?
A bar chart is a graphical representation of data using rectangular bars to compare values across different categories.

Which function is used to create a bar chart in Python?
The plt.bar() function from the Matplotlib library is used to create a bar chart.


#Q2
  import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv('student.csv')
df = df.dropna()
le = LabelEncoder()
df['Performance'] = le.fit_transform(df['Performance'])
X = df[['Study_Hours', 'Attendance', 'Marks']]
y = df['Performance']
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
print("Confusion Matrix:")

#csv
Student_ID,Study_Hours,Attendance,Marks,Performance
1,2,60,45,Low
2,3,65,50,Low

#viva
What is KNN?
KNN (K-Nearest Neighbors) is a supervised machine learning algorithm used to classify students into different performance categories.

What is a confusion matrix?
A confusion matrix is a table used to evaluate a classification model by showing the number of correct and incorrect predictions.



print(confusion_matrix(y_test, y_pred))
