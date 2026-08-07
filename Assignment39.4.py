import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay


df = pd.read_csv("student_performance_ml.csv")
model = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Expected Answers:")
print(y_test)
print("Predicted Answers:")
print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of model is:{accuracy*100}%")
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]
print("Confusion Matrix:")
print(cm)
print("Explanation:")
print(f"True Positive (TP): {TP}")
print("Actual = Pass (1), Predicted = Pass (1)")
print(f"True Negative (TN): {TN}")
print("Actual = Fail (0), Predicted = Fail (0)")
print(f"False Positive (FP): {FP}")
print("Actual = Fail (0), Predicted = Pass (1)")
print("Model incorrectly predicted a failing student as Pass.")
print(f"False Negative (FN): {FN}")
print("Actual = Pass (1), Predicted = Fail (0)")
print("Model incorrectly predicted a passing student as Fail.")