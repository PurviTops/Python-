#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 18]

plt.plot(x, y, marker='o', linestyle='--', color='green', label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
tips = sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.title('Total Bill vs Tip')
plt.show()


# In[3]:


# Histogram
plt.hist(tips["total_bill"], bins=10, color='skyblue')
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# Bar chart
tips.groupby("day")["tip"].mean().plot(kind='bar', color='orange')
plt.title('Average Tip by Day')
plt.ylabel('Average Tip')
plt.show()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix heatmap
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[5]:


# Box plot
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title('Boxplot of Total Bill by Day')
plt.show()

# Violin plot
sns.violinplot(x="day", y="total_bill", data=tips)
plt.title('Violin Plot of Total Bill by Day')
plt.show()


# In[6]:


# Pair plot
sns.pairplot(tips, hue="sex")
plt.show()

# Joint plot
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")
plt.show()


# In[7]:



import pandas as pd

df = sns.load_dataset("iris")


print("Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Basic stats:\n", df.describe())

sns.pairplot(df, hue='species')
plt.show()


# In[ ]:




