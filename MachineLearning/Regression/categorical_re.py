import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df= pd.read_csv('carprices.csv')
df.columns = df.columns.str.replace(" ","_")
dummy = pd.get_dummies(df["Car_Model"]).astype(int)
con_df = pd.concat([df,dummy],axis='columns')
con_df.drop(['Car_Model','BMW X5'],axis='columns',inplace=True)

x=con_df.drop('Sell_Price($)',axis='columns')
# print(x,x.shape)
y=con_df['Sell_Price($)']
# print(y,y.shape)
model = LinearRegression()
model.fit(x, y)
print("Model training completed.")
print(model.score(x,y))

test=np.array([[22500, 2, 0, 0]])
predicted_price = model.predict(test)
y_pred = model.predict(x)
print("Predicted price for a car with 22500 miles and 2 years old:", predicted_price)


plt.scatter(con_df["Mileage"], con_df["Sell_Price($)"], color='blue', label='Actual Price')
plt.scatter(con_df["Mileage"], y_pred, color='red', label='Predicted Price')

plt.xlabel('Mileage')
plt.ylabel('Sell Price ($)')
plt.title('Car Price Prediction Based on Mileage')
plt.legend()
plt.show()


# # print(con_df.head())
# print(con_df)


