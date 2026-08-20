import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    # ============================================================
    # Q1 -> Create a DataFrame and print basic information
    # ============================================================

    data = {
        "Name": ["Amit", "Sagar", "Pooja"],
        "Math": [85, 90, 78],
        "Science": [92, 88, 80],
        "English": [75, 80, 82]
    }

    df = pd.DataFrame(data)

    # Print shape, columns and data types
    print("Shape of dataframe:")
    print(df.shape)

    print("Columns are:")
    print(df.columns)

    print("Data types are:")
    print(df.dtypes)


    # ============================================================
    # Q2 -> Print descriptive statistics using .describe()
    # ============================================================

    print("Detailed description:")
    print(df.describe())


    # ============================================================
    # Q3 -> Add Total column as sum of all subject marks
    # ============================================================

    df["Total"] = 0

    # Loop Based Approach
    for i in range(3):
        total = (
            df["Math"][i]
            + df["Science"][i]
            + df["English"][i]
        )

        df.loc[i, "Total"] = total

    # Without loop:
    # df["Total"] = df["Math"] + df["Science"] + df["English"]

    print("After adding column Total:")
    print(df)


    # ============================================================
    # Q4 -> Display students who scored more than 85 in Science
    # ============================================================

    print("Students who have more than 85 marks in Science:")
    print(df.loc[df["Science"] > 85])


    # ============================================================
    # Q5 -> Replace 'Pooja' with 'Puja' in the Name column
    # ============================================================

    df.loc[df["Name"] == "Pooja", "Name"] = "Puja"

    print("After changing the name from Pooja to Puja:")
    print(df)


    # ============================================================
    # Q6 -> Sort DataFrame by Total marks in descending order
    # ============================================================

    df = df.sort_values("Total", ascending=False)

    print("After sorting in descending order by Total column:")
    print(df)


    # ============================================================
    # Q7 -> Create a bar plot of student names vs Total marks
    # ============================================================

    plt.title("Students vs Marks Bar Plot")

    plt.bar(
        df["Name"],
        df["Total"],
        label="Total Marks",
        width=0.6
    )

    plt.xlabel("Name of Students")
    plt.ylabel("Total Marks")
    plt.legend(loc="upper right")

    plt.show()


    # ============================================================
    # Q8 -> Plot a line chart of Amit's marks across all subjects
    # ============================================================

    plt.title("Amit's Marks")

    x = ["Math", "Science", "English"]

    y = df.loc[
        df["Name"] == "Amit",
        ["Math", "Science", "English"]
    ].iloc[0]

    plt.plot(x, y, marker="o", label="Marks")

    plt.xlabel("Subjects")
    plt.ylabel("Marks of Amit in each subject")
    plt.legend(loc="upper right")

    plt.show()


    # ============================================================
    # Q9 -> Create DataFrame with missing values and fill them
    #        using the mean of their respective columns
    # ============================================================

    data2 = {
        "Name": ["Amit", "Sagar", "Pooja"],
        "Math": [np.nan, 76, 88],
        "Science": [91, np.nan, 85]
    }

    df2 = pd.DataFrame(data2)

    print("DataFrame before removing NA values:")
    print(df2)

    # Replace missing values with the mean of the respective column
    df2 = df2.fillna(df2.mean(numeric_only=True))

    print("DataFrame after filling NA values:")
    print(df2)


    # ============================================================
    # Q10 -> Drop the English column from the original DataFrame
    # ============================================================

    df = df.drop(columns=["English"])

    print("Original DataFrame after dropping English column:")
    print(df)


if __name__ == "__main__":
    main()