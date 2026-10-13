#Q1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Pune', 'Mumbai'],
    'Membership_Type': ['Gold', 'Silver', 'Platinum', 'Gold', 'Silver']
}
customer_DataFrame = pd.DataFrame(data)
print("Before Encoding:")
print(customer_DataFrame)
le = LabelEncoder()
for col in ['Gender', 'City', 'Membership_Type']:
    customer_DataFrame[col] = le.fit_transform(customer_DataFrame[col])
print("\nAfter Encoding:")
print(customer_DataFrame)

#viva
What is Label Encoding?
Label Encoding is a technique that converts categorical data into numerical values.

Why is Label Encoding used?
It converts text categories into numbers so that machine learning algorithms can process them.


#Q2
  import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Netflix': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Amazon_Prime': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)
print("Customer DataFrame:")
print(df)
netflix = set(df[df['Netflix'] == 'Yes']['Customer_ID'])
amazon = set(df[df['Amazon_Prime'] == 'Yes']['Customer_ID'])
only_netflix = len(netflix - amazon)
only_amazon = len(amazon - netflix)
both = len(netflix & amazon)
plt.figure(figsize=(6, 5))
venn2(
    subsets=(only_netflix, only_amazon, both),
    set_labels=('Netflix', 'Amazon Prime'),
    set_colors=('red', 'blue'),
    alpha=0.6
)
plt.title("Netflix and Amazon Prime Users")
plt.show()
labels = ['Only Netflix', 'Only Amazon Prime', 'Both Services']
sizes = [only_netflix, only_amazon, both]
colors = ['red', 'blue', 'green']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90)
plt.title("Customer Subscription Distribution")
plt.show()

#viva
What is a Venn diagram?
A Venn diagram is a visual representation that shows the similarities and differences between two or more sets.

Why is a pie chart used in this program?
A pie chart represents the percentage of customers using only Netflix, only Amazon Prime, and both services.

