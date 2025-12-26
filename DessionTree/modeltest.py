import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df= pd.read_csv("salaries.csv")

x=df.drop('salary_more_then_100k',axis='columns')
y=df['salary_more_then_100k']


# le=LabelEncoder()
# for col in x.columns:
#     if x[col].dtype=='object':
#         x[col]=le.fit_transform(x[col]) 

le_company =LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

x['company']=le_company.fit_transform(x['company'])
x['job']=le_job.fit_transform(x['job']) 
x['degree']=le_degree.fit_transform(x['degree'])

print(x.head())
model=tree.DecisionTreeClassifier()
model.fit(x,y)

# tree.plot_tree(model)
# plt.show()
print("Score",model.score(x,y))
print(model.predict([[2,1,0]]))
