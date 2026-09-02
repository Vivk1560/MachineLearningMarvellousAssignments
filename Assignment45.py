# ============================================================================
# Assignment 45
# ============================================================================
# Topic: Data Processing, Preprocessing and Visualization using Pandas,
#        Scikit-Learn and Matplotlib
#
# Description:
# This assignment demonstrates various data preprocessing and analysis
# techniques using a student marks dataset. The tasks include Min-Max Scaling,
# One-Hot Encoding, GroupBy operations, conditional columns, data export,
# and basic data visualization.
#
# Libraries Used:
# - Pandas
# - NumPy
# - Scikit-Learn
# - Matplotlib
# ============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ============================================================================
# Dataset
# ============================================================================

data = {
    "Name": ["Amit", "Sagar", "Pooja"],
    "Math": [85, 90, 78],
    "Science": [92, 88, 80],
    "English": [75, 80, 82]
}

df = pd.DataFrame(data)

print("Initial Student Data:")
print(df)


# ============================================================================
# Q1 -> Normalize the Math Scores using Min-Max Scaling
# ============================================================================

print("\n" + "=" * 70)
print("Q1: Min-Max Scaling of Math Scores")
print("=" * 70)

# Manual Min-Max Scaling using Pandas
df["Math_Normalized_Manual"] = (
    (df["Math"] - df["Math"].min())
    / (df["Math"].max() - df["Math"].min())
)

# Min-Max Scaling using Scikit-Learn
scaler = MinMaxScaler()

df["Math_Normalized_Sklearn"] = scaler.fit_transform(
    df[["Math"]]
)

print("Math scores normalized using both methods:")
print(
    df[
        [
            "Name",
            "Math",
            "Math_Normalized_Manual",
            "Math_Normalized_Sklearn"
        ]
    ]
)


# ============================================================================
# Q2 -> Create a Gender column and perform One-Hot Encoding
# ============================================================================

print("\n" + "=" * 70)
print("Q2: One-Hot Encoding of Gender")
print("=" * 70)

# Create Gender column
df["Gender"] = ["Male", "Male", "Female"]

print("Data after adding Gender column:")
print(df)


# --------------------------------------------------------------------------
# Method 1 -> One-Hot Encoding using Pandas
# --------------------------------------------------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=["Gender"],
    dtype=int
)

print("\nOne-Hot Encoding using Pandas:")
print(df_encoded)


# --------------------------------------------------------------------------
# Method 2 -> One-Hot Encoding using Scikit-Learn
# --------------------------------------------------------------------------

# Create an independent copy for the Scikit-Learn method
df2 = df.copy()

encoder = OneHotEncoder(sparse_output=False)

enc = encoder.fit_transform(
    df2[["Gender"]]
)

# Convert encoded NumPy array into a DataFrame
df_new = pd.DataFrame(
    enc,
    columns=encoder.get_feature_names_out(["Gender"])
)

# Remove original categorical column
df2 = df2.drop(columns=["Gender"])

# Add encoded columns to the DataFrame
df2 = pd.concat(
    [df2, df_new],
    axis=1
)

print("\nOne-Hot Encoding using Scikit-Learn:")
print(df2)


# ============================================================================
# Q3 -> Group Students by Gender and Calculate Average Marks
# ============================================================================

print("\n" + "=" * 70)
print("Q3: Average Marks by Gender")
print("=" * 70)

print("\nAverage Math Marks by Gender:")
print(df.groupby("Gender")["Math"].mean())

print("\nAverage Science Marks by Gender:")
print(df.groupby("Gender")["Science"].mean())

print("\nAverage English Marks by Gender:")
print(df.groupby("Gender")["English"].mean())


# ============================================================================
# Q4 -> Plot a Pie Chart of Subject Marks for Sagar
# ============================================================================

print("\n" + "=" * 70)
print("Q4: Pie Chart of Sagar's Subject Marks")
print("=" * 70)

# Select Sagar's marks for the three subjects
values = df.loc[
    df["Name"] == "Sagar",
    ["Math", "Science", "English"]
].values[0]

labels = ["Math", "Science", "English"]

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%"
)

plt.title("Sagar's Subject Marks")
plt.axis("equal")
plt.show()


# ============================================================================
# Q5 -> Create a Status Column
# ============================================================================

print("\n" + "=" * 70)
print("Q5: Student Status Based on Total Marks")
print("=" * 70)

# Total marks >= 250 -> Pass
# Otherwise -> Fail
df["Status"] = np.where(
    (
        df["Math"]
        + df["Science"]
        + df["English"]
    ) >= 250,
    "Pass",
    "Fail"
)

print("Data after adding Status column:")
print(df)


# ============================================================================
# Q6 -> Count How Many Students Passed
# ============================================================================

print("\n" + "=" * 70)
print("Q6: Number of Students Passed")
print("=" * 70)

pass_count = len(
    df.loc[df["Status"] == "Pass"]
)

print("Number of students passed:", pass_count)


# ============================================================================
# Q7 -> Export the Final DataFrame to CSV
# ============================================================================

print("\n" + "=" * 70)
print("Q7: Export DataFrame to CSV")
print("=" * 70)

df.to_csv(
    "students_processed.csv",
    index=False
)

print("DataFrame exported successfully to students_processed.csv")


# ============================================================================
# Q8 -> Plot a Histogram for Math Marks
# ============================================================================

print("\n" + "=" * 70)
print("Q8: Histogram of Math Marks")
print("=" * 70)

plt.hist(
    df["Math"],
    bins=5
)

plt.title("Math Mark Distribution")
plt.xlabel("Math Marks")
plt.ylabel("Number of Students")
plt.show()


# ============================================================================
# Q9 -> Rename Column Math to Mathematics
# ============================================================================

print("\n" + "=" * 70)
print("Q9: Rename Math Column")
print("=" * 70)

df = df.rename(
    columns={
        "Math": "Mathematics"
    }
)

print("DataFrame after renaming 'Math' to 'Mathematics':")
print(df)


# ============================================================================
# Q10 -> Plot a Boxplot for English Marks
# ============================================================================

print("\n" + "=" * 70)
print("Q10: Boxplot of English Marks")
print("=" * 70)

plt.boxplot(
    df["English"]
)

plt.title("English Marks Distribution")
plt.ylabel("English Marks")
plt.show()

