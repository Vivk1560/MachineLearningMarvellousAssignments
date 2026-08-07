import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
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
plt.figure(figsize=(10, 6))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Visualization")
plt.show()