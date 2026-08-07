import pandas as pd 

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
print("First 5 Records From The Dataset Are:")
print(df.head()) #First 5 Records
print("Last 5 Records From The Dataset Are:")
print(df.tail()) #Last 5 Records
print("Number of Rows And Columns Are:",df.shape) #Number of Rows And Columns
print("Names of columns are:",list(df.columns)) #Names of columns
print("Data Types Of Each Columns Are:")
print(df.dtypes) #Data types of each columns