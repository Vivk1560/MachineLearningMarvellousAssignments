from OOPKnn import KNeighboursClassifier

def main():
    df = [
        {"point":"A","co-ordinates":(1,2),"label":"Red"},
        {"point":"B","co-ordinates":(2,3),"label":"Red"},
        {"point":"C","co-ordinates":(3,1),"label":"Blue"},
        {"point":"D","co-ordinates":(6,5),"label":"Blue"},
    ]
    model = KNeighboursClassifier(k=4)
    model = model.fit(df)
    x = int(input("Enter the x coordinate for new point: "))
    y = int(input("Enter the y coordinate for new point: "))
    newP = {"co-ordinates":(x,y)}
    ans,nearest = model.predict(newP)
    print("Nearest Neighbors")
    for i in nearest:
        dist = i['distanceWithNewPoint']
        print(f"{i['point']} - Distance: {dist:.2f}")
    print(f"Predicted Class: {ans}")


if __name__ == "__main__":
    main()
