#Q1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    'City': ['Mumbai', 'Pune', 'Delhi', 'Nashik', 'Chennai'],
    'House_Type': ['Flat', 'Villa', 'Flat', 'Bungalow', 'Villa'],
    'House_Area': [1000, 1500, 1200, 800, 1800],
    'Number_of_Rooms': [2, 3, 2, 1, 4],
    'Monthly_Rent': [20000, 35000, 25000, 12000, 40000]
})

print("Original Data:")
print(df)

cols = ['House_Area', 'Number_of_Rooms', 'Monthly_Rent']

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

print("\nNormalized Data:")
print(df)

#Q2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

df = pd.DataFrame({
    'Customer': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Netflix': [1, 1, 1, 0, 0, 1],
    'Amazon_Prime': [0, 1, 0, 1, 1, 1]
})

netflix = set(df[df['Netflix'] == 1]['Customer'])
amazon = set(df[df['Amazon_Prime'] == 1]['Customer'])

only_netflix = len(netflix - amazon)
only_amazon = len(amazon - netflix)
both = len(netflix & amazon)

plt.figure(figsize=(6, 5))
venn2([netflix, amazon], set_labels=('Netflix', 'Amazon Prime'))
plt.title("Netflix and Amazon Prime Customers")
plt.show()

labels = ['Only Netflix', 'Only Amazon Prime', 'Both Services']
values = [only_netflix, only_amazon, both]

plt.pie(values, labels=labels, autopct='%1.1f%%',
        colors=['red', 'orange', 'skyblue'], startangle=90)
plt.title("Customer Subscription Distribution")
plt.show()

print("Only Netflix:", only_netflix)
print("Only Amazon Prime:", only_amazon)
print("Both Services:", both)
