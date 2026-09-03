from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd

data = {
    "StudyHours":[1,2,3,4,5],
    "Marks":[50,55,60,65,70]
}
df = pd.DataFrame(data=data)
X = df[["StudyHours"]]
Y = df["Marks"]
model = LinearRegression()
model = model.fit(X,Y)
print("Coefficient of Regression is",model.coef_[0])
print("Intercept is:",model.intercept_)