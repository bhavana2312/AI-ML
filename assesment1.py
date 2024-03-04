import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/Copy of Dataset - Titanic.csv')
df

print(df.shape)

# Get summary statistics of numerical variables
print(df.describe())

# Check the data types of variables
print(df.dtypes)

df.head()

print(df.isnull().sum())

sns.boxplot(x=df['survived'], y=df['fare'])
plt.xlabel('survival status')
plt.ylabel('fare')
plt.title('survival status vs. fare')
plt.show()

sns.countplot(x='survived', data=df)
plt.xlabel('survival status')
plt.ylabel('count')
plt.title('survival count')
plt.show()

plt.hist(df['age'], bins=10)
plt.xlabel('age')
plt.ylabel('frequency')
plt.title('Distribution of Age')
plt.show()

correlation = df[['age', 'fare']].corr()
print(correlation)

cross_tab = pd.crosstab(df['pclass'], df['survived'])
print(cross_tab)

max_thresold=df['pclass'].quantile(0.25)
max_thresold

df[df['pclass']>max_thresold]

min_thresold=df['pclass'].quantile(0.75)
min_thresold

df[df['pclass']<min_thresold]
