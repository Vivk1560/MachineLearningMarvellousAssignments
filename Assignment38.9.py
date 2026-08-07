import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance_ml.csv")
passStudents = df[df["FinalResult"] == 1]
failStudents = df[df["FinalResult"] == 0]
plt.scatter(passStudents["AssignmentsCompleted"],passStudents["FinalResult"],color="green",label="Pass")
plt.scatter(failStudents["AssignmentsCompleted"],failStudents["FinalResult"],color="red",label="Fail")
plt.title("Assignments Completed vs Final Result")
plt.xlabel("Assignments Completed")
plt.ylabel("Final Result")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.legend()
plt.grid(True)
plt.show()