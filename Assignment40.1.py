import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_performance_ml.csv")
model = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)
print("Feature Importance:")
print(importance)
print("Most Important Feature:")
print(f"{importance.idxmax()}={importance.max()}")
print("Least Important Feature:")
print(f"{importance.idxmin()}={importance.min()}")