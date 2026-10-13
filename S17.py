#Q!
import pandas as pd

student_details = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha'],
    'Age': [20, 21, 19, 22]
})
academic_records = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4],
    'Marks': [85, 90, 78, 92],
    'Grade': ['A', 'A+', 'B', 'A+']
})
print("Student Details:")
print(student_details)
print("\nAcademic Records:")
print(academic_records)
integrated_data = pd.merge(student_details, academic_records, on='Student_ID')
print("\nIntegrated Data:")
print(integrated_data)

#Q2
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    'City': ['Mumbai', 'Pune', 'Delhi', 'Chennai', 'Nashik'],
    'Population': [12400000, 3100000, 16700000, 7100000, 1500000],
    'GDP': [310, 69, 167, 78, 15],
    'Literacy_Rate': [89, 91, 88, 90, 90]
})
fig = px.treemap(df, path=['City'], values='Population',
                 title='City-wise Population Distribution',
                 color='Population', color_continuous_scale='Blues')
fig.show()

fig = px.scatter_3d(df, x='Population', y='GDP', z='Literacy_Rate',
                    color='City',
                    title='3D Scatter Plot of Cities',
                    labels={'Population': 'Population',
                            'GDP': 'GDP (in billions)',
                            'Literacy_Rate': 'Literacy Rate (%)'})
fig.show()
