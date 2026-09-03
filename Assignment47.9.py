import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "StudyHours": [1, 2, 3, 4, 5],
    "SleepHours": [7, 6, 7, 6, 8],
    "Marks": [50, 55, 60, 65, 70]
}
df = pd.DataFrame(data=data)
X = df.drop(columns = ["Marks"])
Y = df["Marks"]
model = LinearRegression()
model = model.fit(X,Y)
for features,coefficients in zip(X.columns,model.coef_):
    print(f"Feature : {features} Coefficient : {coefficients}")