import tensorflow as tf
from preprocess import SAMPLE_NUM, TRAIN_NAME, TEST_SIZE


def train(model, data, batch_size=1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    total_loss = 0
    for i in range(0, SAMPLE_NUM * (1-TEST_SIZE), batch_size):  # loop over all training examples we have
        inputs = data[TRAIN_NAME][i:i+batch_size]  # creating a batch of inputs here
        out, mu, logvar = model.call(inputs)
        with tf.GradientTape() as tape:
            loss = model.loss(out, inputs, mu, logvar)
            total_loss += loss
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return total_loss


def main():
    """
    Reads data from chembl22/chembl22.h5, trains model, then tests model!
    :return:
    """
    pass


if __name__ == "__main__":
    main()
