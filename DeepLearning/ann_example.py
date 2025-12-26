import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = {
    "age": [22, 25, 30, 35, 40, 45],
    "experience": [1, 2, 5, 7, 10, 12],
    "salary": [20000, 25000, 40000, 55000, 70000, 85000]
}

df = pd.DataFrame(data)
X = df[["age", "experience"]]
y = df["salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=2, verbose=0)
predictions = model.predict(X_test)
# df.loc[X_test, "predicted_salary"] = predictions.flatten()
print(df)
def predict_salary(age, experience):
    input_data = np.array([[age, experience]])
    input_data = scaler.transform(input_data)
    predicted_salary = model.predict(input_data)
    return predicted_salary[0][0]

# Example usage:
predicted = predict_salary(22, 1)
print(f"Predicted salary for age 22 and experience 1 year: {predicted}")