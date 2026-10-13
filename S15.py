#Q1
import pandas as pd

age_records = pd.DataFrame({
    'Person_ID': [1, 2, 3, 4, 5, 6],
    'Age': [18, 22, 25, 30, 35, 40]
})

print("Original Data:")
print(age_records)

age_records['Age_Bin'] = pd.qcut(age_records['Age'], q=3)

print("\nDiscretized Data:")
print(age_records)


#Q2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.DataFrame({
    'City': ['Mumbai', 'Pune', 'Nashik', 'Delhi', 'Chennai'],
    'Area': [603, 331, 259, 1484, 426],
    'Temperature': [28, 25, 26, 30, 32],
    'Rainfall': [2200, 800, 700, 700, 1400],
    'Population_Density': [20000, 6000, 6000, 12000, 11000]
})

fig = px.treemap(df, path=['City'], values='Population_Density',
                 color='Population_Density',
                 title='Population Density of Cities')
fig.show()

fig = px.scatter_3d(df, x='Area', y='Temperature', z='Rainfall',
                    color='City', title='3D Scatter Plot of Cities',
                    labels={'Area': 'Area (sq. km)',
                            'Temperature': 'Average Temperature (°C)',
                            'Rainfall': 'Rainfall (mm)'})
fig.show()
