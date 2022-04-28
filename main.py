import tensorflow as tf
from preprocess import TRAIN_NAME, TEST_NAME, OUT_FILE
import h5py
from model import Model
import numpy as np


def train_model(model, data, batch_size=1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    total_loss = 0
    for i in range(0, data.shape[0], batch_size):  # loop over all training examples we have
        inputs = data[i:i+batch_size]  # creating a batch of inputs here
        with tf.GradientTape() as tape:
            out, mu, logvar = model.call(inputs)
            loss = model.loss(out, inputs, mu, logvar)
            print(loss)
            total_loss += loss
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return total_loss


def main():
    """
    Reads data from chembl22/chembl22.h5, trains model, then tests model!
    :return:
    """
    data = h5py.File(OUT_FILE)
    train = data[TRAIN_NAME][:]
    test = data[TEST_NAME][:]

    print("Now making model")
    molencoder = Model()

    print("Now training")
    total_loss = train_model(molencoder, train)


if __name__ == "__main__":
    main()
