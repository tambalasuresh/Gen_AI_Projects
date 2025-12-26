import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Size': [1500, 1600, 1700, 1800, 1900],
    "Price": [300000, 320000, 340000, 360000, 380000]
}

df = pd.DataFrame(data)
X = df[['Size']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:\n", X_train)
print("X_test:\n", X_test)  
print("y_train:\n", y_train)
print("y_test:\n", y_test)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
print("After resetting indices:")
print("X_train:\n", X_train)    
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)

X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Data saved to CSV files.")
print("Data splitting completed.")

model = LinearRegression()
model.fit(X, y)
print("Model training completed.")
y_pred = model.predict(X_test)
main_pre = model.predict([[2200]])
print("Predictions on test set:", main_pre)

print("Predicted price for a house of size 2200 sq ft:", main_pre[0])
print("Predictions on test set:", y_pred)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)') 
plt.title('House Price Prediction')
plt.legend()
plt.show()