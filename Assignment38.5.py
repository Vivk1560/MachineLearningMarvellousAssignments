import pandas as pd

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
avgStudyHrs = df.groupby("FinalResult")["StudyHours"]
print("Average Study Hours of students who are failing:",avgStudyHrs.get_group(0).mean())
print("Average Study Hours of students who are passing:",avgStudyHrs.get_group(1).mean())

#Here we first grouped the dataframe with comparison by finalresult and then extracted the studyhours 
#wala column in the avgStudyHrs
#so avgStudyHrs contains 2 groups as 2 labels are there in the dataset
#now avgstudyhrs has 2 groups i.e. one studyhrs wala group with final result as 0
#one finalresult wala group with final result as 1 and both having rows of the column studyhrs only
#this avg will help us understand if avg studyhrs for passing students is more 
# we get a more avg hrs in passing students and vice versa
#we have a second method to do this as well

print(df[["StudyHours","FinalResult"]].corr())

#This is the second method which results into a 2x2 matrix and when we compare the matrix's
#values and we see correlation between studyhours and finalresult as more than 0 we can see there
#is a correlation between 2 and we can analyze if we see any relation between both columns