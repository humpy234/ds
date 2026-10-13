#Q1
import pandas as pd

book_records = pd.DataFrame({
    'Book_ID': [1, 2, 3, 4, 5, 6],
    'Book_Price': [100, 200, 300, 400, 500, 600]
})

print("Original Data:")
print(book_records)

book_records['Price_Bin'] = pd.cut(book_records['Book_Price'], bins=3)

print("\nDiscretized Data:")
print(book_records)

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

print("World Cities Dataset:")
print(df)

fig = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    hover_name='city',
    size='population',
    color='country',
    title='World Cities Population Map',
    projection='natural earth'
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
