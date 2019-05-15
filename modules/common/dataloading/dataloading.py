import pandas as pd


def load_data(train_file, validation_file, test_file,test_file_B,validation_file_B, sep=',', header=0, **kwargs):
    train = pd.read_csv(train_file, sep=sep, header=header, quoting=1).sample(frac=1).values
    valid = pd.read_csv(validation_file, sep=sep, header=header, quoting=1).sample(frac=1).values
    valid_B = pd.read_csv(validation_file_B, sep=sep, header=header, quoting=1).sample(frac=1).values
    test = pd.read_csv(test_file, sep=sep, header=header, quoting=1).values
    test_B = pd.read_csv(test_file_B, sep=sep, header=header, quoting=1).values

    return train,valid,test,test_B,valid_B
