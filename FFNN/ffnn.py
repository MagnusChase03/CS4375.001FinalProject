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
The first tensor is the training data, and the last one are the labels.
"""
def load_data(filepath: str) -> (tf.Tensor, tf.Tensor):
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

    return (tf.constant(rows[:, :-1]), tf.constant(rows[:, -1]))


"""
Returns the nerual network model to use.
"""
def create_model(learning_rate: float):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation="sigmoid"), # Make function create dynamic layers?
        tf.keras.layers.Dense(6, activation="sigmoid"),
        tf.keras.layers.Dense(2) # Output layer, need to call softmax manually
    ]) 

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        ), 
        loss=loss_fn, 
        metrics=["accuracy"])

    return model


"""
Feeds data and labels into the network for training
"""
def forward(model, data: tf.Tensor, labels: tf.Tensor, batch_size: int, epochs: int):
    model.fit(x=data, y=labels, epochs=epochs, shuffle=True, validation_split=0.3, batch_size=batch_size)


"""
Program Entrypoint
"""
def main():
    tensor_data = load_data("./data.csv")
    model = create_model(0.01)
    forward(model, tensor_data[0], tensor_data[1], 5, 10)


main()
