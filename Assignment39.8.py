import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay


df = pd.read_csv("student_performance_ml.csv")
print("Shape of dataset:",df.shape)
print("Column Names:",list(df.columns))
print("Missing Values Per Column:",df.isnull().sum())
print("Class Distribution")
print(df["FinalResult"].value_counts())
print("Statistical Report Of Dataset :")
print(df.describe())
plt.figure(figsize=(6,5))
plt.scatter(df["StudyHours"],
            df["PreviousScore"],
            c=df["FinalResult"],
            cmap="bwr",
            s=80)

plt.xlabel("Study Hours")
plt.ylabel("Previous Score")
plt.title("Study Hours vs Previous Score")
plt.colorbar(label="Final Result")
plt.show()
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
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