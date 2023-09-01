
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def handleCSV(csvPath, classificationColumn, testTrainingRatio):

    pd.set_option('display.float_format', '{:.2f}'.format)

    dataframe = pd.read_csv(csvPath, header=None)

    headers = dataframe.iloc[0].tolist()
    dataframe.columns = headers
    dataframe.drop(index=dataframe.index[0], axis=0, inplace=True)

    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    features = dataframe.drop(columns=[classificationColumn])
    classifier = dataframe[[classificationColumn]]

    x_train, x_test, y_train, y_test = train_test_split(features, classifier, test_size=testTrainingRatio)
    
    x_train = [array.reshape(-1, 1) for array in x_train.to_numpy()]
    x_test = [array.reshape(-1, 1) for array in x_test.to_numpy()]
    y_train = [array.reshape(-1, 1) for array in y_train.to_numpy()]
    y_test = [array.reshape(-1, 1) for array in y_test.to_numpy()]

    return x_train, x_test, y_train, y_test

def reshapeData(data):
    return [array.reshape(-1, 1) for array in data]