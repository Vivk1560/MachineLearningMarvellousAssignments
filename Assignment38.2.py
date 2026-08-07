import pandas as pd

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
print("Total number of students in the dataset are:",df.shape[0])
count = df["FinalResult"].value_counts()
print("Total Number of Failed Students which is FinalResult as 0 are:",count[0])
print("Total Number of Passed Students which is FinalResult as 1 are:",count[1])