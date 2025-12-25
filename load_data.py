import pandas as pd 

df = pd.read_csv("customer.csv")
df.info()
df.tail()
df.head()
df.shape
df.columns

df.groupby("city")["salary"].mean()
df['chunk'].value_counts()


df.isna().sum()
df['age'].fillna(df["age"].mean())
df["salary"].fillna(df["salary"].median())