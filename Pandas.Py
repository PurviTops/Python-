
import pandas as pd
import numpy as np

#1 & 2
colors = pd.Series(["Red", "Blue", "Green"])
print(colors)

#3
car_types = pd.Series(["Sedan", "SUV", "Truck"])
print(car_types)

#4
car_data = pd.DataFrame({"Car Type": car_types, "Color": colors})
print(car_data)

#5
print(car_data.dtypes)

#6
print(car_data.describe())

#7
print(car_data.info())

#8
numbers = pd.Series([10, 20, 30, 40, 50])
print("Mean:", numbers.mean())

#9
print("Sum:", numbers.sum())

#10
print(car_data.columns)

#11
print("Length of DataFrame:", len(car_data))

#12
print(car_data.head(5))

#13
print(car_data.head(7))

#14
print(car_data.tail(5))

#15
car_data_expanded = pd.DataFrame({
    "Car Type": ["Sedan", "SUV", "Truck", "Hatchback", "Convertible"],
    "Color": ["Red", "Blue", "Green", "Black", "White"]
})
print(car_data_expanded.loc[3])

#16
print(car_data_expanded.iloc[3])

