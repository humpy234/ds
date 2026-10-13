#Q1
import pandas as pd
data = {
    'Person_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [12, 18, 25, 32, 38, 45, 52, 60, 68, 75]
}
age_records = pd.DataFrame(data)
print("Original Data:")
print(age_records)

age_records['Age_Group'] = pd.cut(
    age_records['Age'],
    bins=3,
    labels=['Young', 'Middle-aged', 'Senior']
)
print("\nDiscretized Data:")
print(age_records)

#viva
1)What is Data Discretization?
Data discretization is the process of converting continuous numerical data into a limited number of intervals or categories.

2)What is Equal Width Binning?
Equal Width Binning divides the range of values into intervals of equal size using the formula: Bin Width = (Maximum Value − Minimum Value) / Number of Bins

#Q2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('youtube_data.csv')
print("YouTube Dataset:")
print(df)
X = df[['Likes']]
y = df['Views']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("\nPredicted Views:")
print(y_pred)
print("\nR2 Score:", r2_score(y, y_pred))
plt.scatter(X['Likes'], y, color='blue', label='Actual Views')
plt.plot(X['Likes'], y_pred, color='red', label='Regression Line')
plt.xlabel('Likes')
plt.ylabel('Views')
plt.title('Linear Regression: YouTube Likes vs Views')
plt.legend()
plt.show()

#csv file 
Video_ID,Likes,Views
1,100,1000
2,200,2000
