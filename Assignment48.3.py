from UserDefinedSimpleLinearRegression import UserDefinedSimpleLinearRegression

import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5]

Y = [20000, 25000, 30000, 35000, 40000]

model = UserDefinedSimpleLinearRegression()

model = model.fit(X, Y)

# Predict salary for 6 years of experience
Y_pred_6 = model.predict([6])

print("Predicted Salary for 6 Years Experience: ₹", Y_pred_6[0])

# Predict salary for training data
Y_pred = model.predict(X)

# Plot data points
plt.scatter(X, Y, label="Data Points")

# Plot regression line
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")

plt.legend()
plt.show()