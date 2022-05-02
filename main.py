import tensorflow as tf
from preprocess import DICT_NAME, TRAIN_NAME, TEST_NAME, OUT_FILE, un_encode, one_hot_smile, pad_smile
import h5py
from model import Model
import numpy as np


def train_model(model, data, batch_size=1000):
    switch = True
    optimizer_max = tf.keras.optimizers.Adam(learning_rate=0.01)
    optimizer_min = tf.keras.optimizers.Adam(learning_rate=0.001)
    total_loss = 0
    for i in range(0, data.shape[0] - 350000, batch_size):  # loop over all training examples we have
        inputs = data[i:i+batch_size]  # creating a batch of inputs here
        with tf.GradientTape() as tape:
            out, mu, logvar = model.call(inputs)
            loss = model.loss(out, inputs, mu, logvar)
            print("Batch " + str(i) + " loss: " + str(float(loss)))
            total_loss += loss
        gradient = tape.gradient(loss, model.trainable_variables)
        if i % 10000 == 0 and i != 0:  # switch LR every 10k samples; "cyclical learning rate" to avoid local min
            switch = not switch
            print("Changing learning rate...")
        if switch:
            optimizer_max.apply_gradients(zip(gradient, model.trainable_variables))
        else:
            optimizer_min.apply_gradients(zip(gradient, model.trainable_variables))

    return total_loss


def generate_molecules(model, character_dict, smile):  # this acts as our test function, as specified in devpost
    """
    Takes in a smile string, then outputs a molecule similar to it by sampling from a learned distribution.
    :param model: TRAINED model, pretty self-explanatory
    :param character_dict: dictionary of character in training set
    :param smile: smile string that we want to use as a base molecule
    :return: smile string of similar molecule generated by our trained model
    """
    one_hot = one_hot_smile(pad_smile(smile), character_dict, preprocess=False)
    one_reshape = np.repeat(one_hot, 1000)  # need this to be compatible with linear layers
    reshape = tf.reshape(one_reshape, [1000, 80, 52])
    output, _, _ = model.call(reshape)  # select the first output of linear layers; they're all the same
    distribution = output[0]

    new_smile = ""
    for i in range(len(distribution)):
        target = distribution[i]  # gets appropriate distribution amongst characters
        probabilities = create_relative_probabilities(target)
        sampled_char_idx = np.random.choice(np.arange(len(character_dict)), p=probabilities)  # samples from dist
        new_smile += character_dict[sampled_char_idx].decode('utf-8')
    return new_smile


def create_relative_probabilities(char_dist):
    """
    Bootleg fix to softmax running over the wrong thing in the decoder. Given some data, it normalizes it to it's
    all proportional to the original data, but sums up to one (for input into random sampling in generate
    molecule). Does this by finding the total sum, then creating a list where each value is = value / total, or its
    % capitalization on the total data.
    :param char_dist: (smile_length, dict_lengh) output from the model with data on a character distribution
    :return: list of probabilities of each character
    """
    total = np.sum(char_dist)
    proportion_list = list()
    for value in char_dist:
        proportion = float(value / total)
        proportion_list.append(round(proportion, 3))  # rounds proportion to 4 decimals; make convergence to 1 easier

    # BEGIN SUM TO 1 CORRECTION HERE
    post_sum = np.sum(proportion_list[:-1])
    difference = 1 - post_sum
    if difference > 0:
        proportion_list[-1] = difference
    else:  # cannot have negative numbers in probability list!
        for prob in proportion_list:
            if prob > abs(difference):
                prob += difference
    # END SUM TO 1 CORRECTION HERE

    return proportion_list


def main():
    """
    Reads data from chembl22/chembl22.h5, trains model, then tests model!
    :return:
    """
    data = h5py.File(OUT_FILE)
    train = data[TRAIN_NAME][:]
    test = data[TEST_NAME][:]
    char_dict = list(data[DICT_NAME][:])

    print("Making model...")
    molencoder = Model()

    print("Training...")
    total_loss = train_model(molencoder, train)

    print("Generating similar molecule...")
    new_mol = generate_molecules(molencoder, char_dict, "HC(H)=C(H)(H)")
    print("New Molecule: " + new_mol)
    new_mol = generate_molecules(molencoder, char_dict, "CC")
    print("New Molecule: " + new_mol)
    new_mol = generate_molecules(molencoder, char_dict, "CC(C)(C)CC")
    print("New Molecule: " + new_mol)
    new_mol = generate_molecules(molencoder, char_dict, "CC(CC)C")
    print("New Molecule: " + new_mol)


if __name__ == "__main__":
    main()
