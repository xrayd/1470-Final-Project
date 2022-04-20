import pandas as pd
from sklearn.model_selection import train_test_split as split

# Hyperparameters for dataset processing
# TODO: make these parameters in preprocess method eventually
MAX_SMILE_LENGTH = 80
SAMPLE_NUM = 750_000  # we have a total of 1,678,393 molecules available
SMILES_COL_NAME = 'canonical_smiles'  # this is default column name
TEST_SIZE = 0.2


# dataset from: ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz
def preprocess():
    chembl22 = pd.read_table('data/chembl_22_chemreps.txt.gz', compression='gzip')
    within_length = chembl22[SMILES_COL_NAME].map(len) <= MAX_SMILE_LENGTH  # list of true/false if under length
    chembl22 = chembl22[within_length].sample(n=SAMPLE_NUM)
    smiles_strings = chembl22[SMILES_COL_NAME]
    train_smiles, test_smiles = split(smiles_strings, shuffle=True, test_size=TEST_SIZE)
    return train_smiles, test_smiles


preprocess()
