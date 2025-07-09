# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# %%
# Load the dataset
data = pd.read_csv('troop_movements.csv')

counts_of_data = data.groupby(['empire_or_resistance']).size().reset_index(name='count')
print(counts_of_data)

# %%
count_of_homeworld = data.groupby('homeworld').size().reset_index(name='count')
print(count_of_homeworld)

# %%
# Groups of unit types

unit_types = data.groupby('unit_type').size().reset_index(name='count')
unit_types

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot for 'empire_or_resistance'
sns.countplot(data=data, x='empire_or_resistance', palette=['#1f77b4', '#ff7f0e'])

plt.title('Counts by Empire or Resistance')
plt.xlabel('Empire or Resistance')
plt.ylabel('Count')
plt.show()


# %%
# Descion Tree that predicts if a character is joining eithe rthe empire or the resistance 
# based on their homeworld and unit type

from sklearn.tree import DecisionTreeClassifier

features = ['unit_type', 'homeworld']
x, y = data[features], data['empire_or_resistance']

# Encode categorical features
x_encoded = pd.get_dummies(x, columns = ['unit_type', 'homeworld'])

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=.3, random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)

y_pred = dtc.predict(x_test)
y_pred


# %%
