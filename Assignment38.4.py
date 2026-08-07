import pandas as pd

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
count = df["FinalResult"].value_counts()
records = df.shape[0]
passPercent = (count[1]/records)*100
failPercent = (count[0]/records)*100
print("Distribution Of Final Result:")
print(count)
print("Percentage of students passing:",passPercent)
print("Percentage of students failing:",failPercent)

#As student's passing percentage is 60 and failing percentage os 40, dataset is a bit imbalanced, 50-50 will be a perfectly balanced dataset, as here the difference is of 20 percent dataset is not fully imbalanced but a bit imbalanced
