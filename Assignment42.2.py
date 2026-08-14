from OOPKnn import KNeighboursClassifier


def main():

    trainingPoints = [
        {"point":"A", "co-ordinates":(1,2), "label":"Red"},
        {"point":"B", "co-ordinates":(2,3), "label":"Red"},
        {"point":"C", "co-ordinates":(3,1), "label":"Blue"},
        {"point":"D", "co-ordinates":(6,5), "label":"Blue"},
        {"point":"E", "co-ordinates":(7,4), "label":"Blue"}
    ]
    newPoint = {"co-ordinates":(4,3)}
    print("Prediction Results")
    for k in [1, 3, 5]:
        model = KNeighboursClassifier(k)
        model.fit(trainingPoints)
        prediction, nearest = model.predict(newPoint)
        print(f"K = {k} → {prediction}")

if __name__ == "__main__":
    main()