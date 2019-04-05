import pandas as pd
import numpy as np

def save_predictions(name, predictions, original_data):
    original_data[:, 2] = predictions
    dataframe = pd.DataFrame(data=original_data)
    dataframe.to_csv(name + '.csv',sep=',', header=False, index=False, quoting=2)

def save_predictions_with_probabilities(name, predictions, original_data, labels, probabilities):
    dataframe = pd.DataFrame(data=original_data[:,:2])
    dataframe.insert(2,2,probabilities[:,0])
    dataframe.insert(3,3,probabilities[:,1])
    dataframe.insert(4,4,predictions)
    dataframe.insert(5,5,labels)
    dataframe.to_csv(name + '.csv',sep=',', header=False, index=False, quoting=2)