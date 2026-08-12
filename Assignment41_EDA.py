import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("WinePredictor.csv")
    features = [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline"
        ]
    df[features].hist(figsize=(15,12),bins=15)
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        axes[row, col].boxplot([
            df[df["Class"] == 1][feature],
            df[df["Class"] == 2][feature],
            df[df["Class"] == 3][feature]
        ])

        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel("Class")
        axes[row, col].set_ylabel(feature)
    plt.tight_layout()
    plt.show()
    class_means = df.groupby("Class")[features].mean()
    print(class_means)
    class_means.T.plot(
    kind="bar",
    figsize=(16, 7)
    )
    plt.title("Mean Feature Values Across Wine Classes")
    plt.xlabel("Features")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    correlation = df[features].corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(
    range(len(features)),
    features,
    rotation=90
    )
    plt.yticks(
    range(len(features)),
    features
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    plt.scatter(
        df[df["Class"]==1]["Flavanoids"],
        df[df["Class"]==1]["Proline"],
        label = "Class1"
    )
    plt.scatter(
            df[df["Class"]==2]["Flavanoids"],
            df[df["Class"]==2]["Proline"],
            label = "Class2"
        )
    plt.scatter(
            df[df["Class"]==3]["Flavanoids"],
            df[df["Class"]==3]["Proline"],
            label = "Class3"
        )
    plt.xlabel("Flavanoids")
    plt.ylabel("Proline")
    plt.title("Flavanoids vs Proline")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()