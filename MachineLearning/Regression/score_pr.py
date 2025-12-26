import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score

data ={
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "test_score": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}

df= pd.DataFrame(data)
x= df[["hours_studied"]]
y = df["test_score"]

model = LinearRegression()
model.fit(x, y)
print("Model training completed.")

test=np.array([[1]])
predicted_score = model.predict(test)
print("Predicted test score for 19 hours studied:", predicted_score[0])
y_pred = model.predict(x)
r2 = r2_score(y, y_pred) 
print("R-squared:", r2)

# plt.figure(figsize=(10,6))
# plt.scatter(x, y, color='blue', label='Actual Data')
# plt.plot(x, model.predict(x), color='red', label='Regression Line')
# plt.xlabel('Hours Studied')
# plt.ylabel('Test Score')
# plt.title('Test Score Prediction Based on Hours Studied')
# plt.legend()    
# plt.show()