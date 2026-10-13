#Q1
import pandas as pd
employee_data = pd.DataFrame({
    'Emp_ID': [1, 2, 3, 4, 4, 6],
    'Age': [25, 30, None, 40, 40, 100],
    'Dept': ['HR', 'IT', 'Sales', 'IT', 'IT', None],
    'Years_Experience': [2, 5, 3, 10, 10, 30],
    'Performance_Rating': [3, 4, 3, 5, 5, 2]
})

print("Original Data:")
print(employee_data)
employee_data.drop_duplicates(inplace=True)
employee_data['Age'] = employee_data['Age'].fillna(employee_data['Age'].mean())
employee_data['Dept'] = employee_data['Dept'].fillna('Unknown')
print("\nCleaned Data:")
print(employee_data)
for col in ['Age', 'Years_Experience', 'Performance_Rating']:
    Q1 = employee_data[col].quantile(0.25)
    Q3 = employee_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = employee_data[(employee_data[col] < lower) | (employee_data[col] > upper)]
    print("\nOutliers in", col, ":")
    print(outliers)

#Q2
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'city': ['Mumbai', 'Delhi', 'London', 'Tokyo', 'New York'],
    'country': ['India', 'India', 'UK', 'Japan', 'USA'],
    'latitude': [19.07, 28.61, 51.50, 35.67, 40.71],
    'longitude': [72.87, 77.20, -0.12, 139.69, -74.00],
    'population': [12400000, 16700000, 8900000, 13900000, 8800000]
})
fig = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    hover_name='city',
    size='population',
    color='country',
    title='World Cities Population Map'
)
fig.show()
fig = px.treemap(
    df,
    path=['country', 'city'],
    values='population',
    title='Population Distribution of World Cities',
    color='population',
    color_continuous_scale='Blues'
)
fig.show()
