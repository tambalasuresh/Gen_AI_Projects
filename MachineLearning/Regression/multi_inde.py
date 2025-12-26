import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data ={
    "sq_fet": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
    "bedroom": [2, 3, 2, 4, 3, 4, 5, 2, 3, 4],
    "age": [5, 4, 6, 3, 7, 2, 8, 1, 9, 0],
    "price": [150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000]
}

df= pd.DataFrame(data)
x=df[["sq_fet", "bedroom", "age"]]
y=df["price"]

model = LinearRegression()
model.fit(x, y)
print("Model training completed.")
test=np.array([[1700, 4, 0]])
predicted_price = model.predict(test)
print("Predicted price for a house with 1700 sq ft, 4 bedrooms, and 0 years old:", predicted_price[0])
y_pred = model.predict(x)       
r2 = r2_score(y, y_pred)
print("R-squared:", r2)

model_coefficients = model.coef_
model_intercept = model.intercept_  
print("Model Coefficients:", model_coefficients)
print("Model Intercept:", model_intercept)

result =  1.00000000e+02 * 1700 + -1.24921881e-12 * 4 + 2.09422658e-13 * 0 + model_intercept
print("Predicted price for a house with 1700 sq ft, 4 bedrooms, and 0 years old:", result)

plt.figure(figsize=(10,6))
plt.scatter(df["sq_fet"], y, color='blue', label='Actual Data')
plt.plot(df["sq_fet"], model.predict(x), color='red', label='Regression Line')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.title('House Price Prediction Based on Square Feet')
plt.legend()
plt.show()