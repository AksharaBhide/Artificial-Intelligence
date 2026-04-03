import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("StudentPerformanceFactors.csv")

# Select columns from your dataset
X = data[['Hours_Studied']]   # independent variable
Y = data['Exam_Score']        # dependent variable

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, Y)

# Predict
Y_pred = model.predict(X)

# Plot
plt.scatter(X, Y, color='blue')
plt.plot(X, Y_pred, color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression - Student Performance")
plt.show()