#==========================================================================================================================
# Assignment : Linear Regression
#
# Objective:
# To implement a Linear Regression model to predict Sales based on advertising expenditure
# on TV, Radio and Newspaper using the Advertising dataset.
#
# Dataset:
# Advertising.csv
#
# Features (Independent Variables):
# 1. TV          - Advertising expenditure on TV
# 2. radio       - Advertising expenditure on Radio
# 3. newspaper   - Advertising expenditure on Newspaper
#
# Target (Dependent Variable):
# sales          - Sales generated from the advertising expenditure
#
# Data Processing:
# - Load and inspect the dataset
# - Remove the unnecessary index column
# - Check for missing values and duplicate records
# - Analyze the correlation between the variables
# - Separate independent and dependent features
# - Split the data into training and testing sets
# - Standardize the independent features using StandardScaler
#
# Machine Learning Algorithm:
# Linear Regression
#
# Model Evaluation:
# - Mean Absolute Error (MAE)
# - Mean Squared Error (MSE)
# - R2 Score
# - Actual vs Predicted Sales visualization using a scatter plot
#
#==========================================================================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------------------------------------------------------
# Step 1 : Load Data
#--------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("Advertising.csv")
print("Initial Entries From The Dataset Are:")
print(df.head())
print()
print("Shape of the dataset is:")
print(df.shape)
print()
print("Description of Dataset is:")
print(df.describe())
print()
print("Information of dataset:")
print(df.info())
print()
print("Columns of dataset:")
print(df.columns)

#--------------------------------------------------------------------------------------------------------------------------
# Step 2 : Process Data
#--------------------------------------------------------------------------------------------------------------------------

df = df.drop(columns=["Unnamed: 0"],axis = 1)
print()
print("After removing column - Unnamed: 0")
print(df.head())
print()
print("Checking for missing values....")
print(df.isna().sum())
print()
print("Number of duplicate records:")
print(df.duplicated().sum())
print()
print("Correlation Matrix is:")
print(df.corr())

#--------------------------------------------------------------------------------------------------------------------------
# Step 3 : Separate Dependent And Independent Features
#--------------------------------------------------------------------------------------------------------------------------

X = df.drop(columns=["sales"])
Y = df["sales"]
print()
print("Independent Features:")
print(X.head())
print()
print("Dependent Features:")
print(Y.head())

#--------------------------------------------------------------------------------------------------------------------------
# Step 4 : Split training and testing data
#--------------------------------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.5)

#--------------------------------------------------------------------------------------------------------------------------
# Step 5 : Scaling the Independent Features
#--------------------------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Step 5 : Model Creation and Training
#--------------------------------------------------------------------------------------------------------------------------

model = LinearRegression()
model = model.fit(X_train,Y_train)

#--------------------------------------------------------------------------------------------------------------------------
# Step 6 : Model Testing
#--------------------------------------------------------------------------------------------------------------------------

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)
mae = mean_absolute_error(Y_test,Y_pred)

#--------------------------------------------------------------------------------------------------------------------------
# Step 7 : Displaying Results
#--------------------------------------------------------------------------------------------------------------------------
print()
for i in range(len(Y_test)):
    print(
        f"Record {i + 1}: "
        f"Predicted = {Y_pred[i]:.2f}, "
        f"Expected = {Y_test.iloc[i]:.2f}"
    )
features = X.columns
print()
print("Coefficients of the model are:")
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")
print()
print("Intercept of the model is:")
print(model.intercept_)

#--------------------------------------------------------------------------------------------------------------------------
# Step 8 : Model Performance Display
#--------------------------------------------------------------------------------------------------------------------------
print()
print("Model Performance:")
print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("R2 Score            :", r2)

#--------------------------------------------------------------------------------------------------------------------------
# Step 9 : Plotting a scatter graph to see the difference
#--------------------------------------------------------------------------------------------------------------------------

plt.scatter(Y_test,Y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot(
    [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    linestyle = "--"
)
plt.show()