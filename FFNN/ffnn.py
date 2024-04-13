import tensorflow as tf
import numpy as np
import csv


"""
Reads file and outputs a list of key-value pairs represented in csv file.
"""
def read_data(filepath: str) -> list[dict]:
    rows = []

    f = open(filepath, "r")
    csvFile = csv.DictReader(f)
    for row in csvFile:
        rows.append(row)
    f.close()

    return rows


"""
Reads data file and returns a list of tensors for the neural network.
"""
def load_data(filepath: str) -> tf.Tensor:
    data = read_data(filepath)
    rows = np.zeros((len(data), len(data[0].keys()) + 3)) # + 3 For one-hot encoding

    xi = 0
    yi = 0
    for row in data:
        for k in row.keys():
            if k == "Sex":
                rows[yi, xi] = 0 if row[k] == "M" else 1
            elif k == "ChestPainType":
                if row[k] == "ATA":
                    rows[yi, xi] = 1
                elif row[k] == "NAP":
                    rows[yi, xi + 1] = 1
                elif row[k] == "ASY":
                    rows[yi, xi + 2] = 1
                else:
                    rows[yi, xi + 3] = 1
                xi += 3
            elif k == "RestingECG":
                rows[yi, xi] = 0 if row[k] == "Normal" else 1
            elif k == "ExerciseAngina":
                rows[yi, xi] = 0 if row[k] == "N" else 1
            elif k == "ST_Slope":
                rows[yi, xi] = 0 if row[k] == "Flat" else 1
            else:
                try:
                    rows[yi, xi] = float(row[k])
                except BaseException:
                    print(f"Failed to convert {row[k]} to float.") 

            xi += 1
        yi += 1
        xi = 0

    return tf.constant(rows)


tensor_data = load_data("./data.csv")
tf.print(tensor_data[0], summarize=-1)
