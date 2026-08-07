import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("student_performance_ml.csv")

model = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
new_students = pd.DataFrame({
    "StudyHours": [3, 6, 5, 8, 4],
    "Attendance": [72, 90, 82, 95, 68],
    "PreviousScore": [55, 70, 62, 88, 50],
    "AssignmentsCompleted": [4, 8, 6, 10, 3],
    "SleepHours": [6, 7, 6, 8, 5]
})
predictions = model.predict(new_students)
new_students["PredictedResult"] = predictions
print(new_students)