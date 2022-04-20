import pandas as pd
from sklearn.model_selection import train_test_split as split
import numpy as np

# Hyperparameters for dataset processing
MAX_SMILE_LENGTH = 80
SAMPLE_NUM = 750_000  # we have a total of 1,678,393 molecules available
SMILES_COL_NAME = 'canonical_smiles'  # this is default column name
TEST_SIZE = 0.2


# dataset from: ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz
# saves dataset to h5 dataset file to prevent running this over and over
def preprocess():
    """
    Reads compressed molecule data and preprocesses it.
    :returns:
    train_indices: numpy array of indices of the training SMILES in smiles_strings
    test_indices: numpy array of indices of the testing SMILES in smiles_strings
    smiles_strings: pandas dataframe of all SMILES strings in our dataset (length defined by hyperparam)
    character_dict: list of all characters present in our dataset
    """
    chembl22 = pd.read_table('data/chembl_22_chemreps.txt.gz', compression='gzip')
    within_length = chembl22[SMILES_COL_NAME].map(len) <= MAX_SMILE_LENGTH  # list of true/false if under length
    chembl22 = chembl22[within_length].sample(n=SAMPLE_NUM)
    smiles_strings = chembl22[SMILES_COL_NAME]
    character_dict = create_chardict(smiles_strings)
    # gets the indices of the SMILES strings for both train and test as Numpy array
    train_indices, test_indices = map(np.array, split(smiles_strings.index, shuffle=True, test_size=TEST_SIZE))

    return train_indices, test_indices, smiles_strings, character_dict


# given a list of strings, creates a dictionary list of all characters in all strings
def create_chardict(strings):
    charset = set()
    for string in strings:
        for char in string:
            charset.add(char)
    charlist = list(charset)
    return charlist


train, test, total, dict = preprocess()
