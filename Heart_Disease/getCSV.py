from fileinput import filename
from foldrm import Classifier
import numpy as np
import pandas as pd

def heart_disease():
    path = "./datasets/"
    fileName = "heart.csv"
    attrs = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'target']
    nums = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'target']
    model = Classifier(attrs=attrs, numeric=nums, label='thal')
    data = model.load_data(path + fileName)
    print('\n% acute dataset', np.shape(data))
    return model, data


def printType():
    path = "./datasets/"
    fileName = "heart.csv"
    temp = pd.read_csv(path+fileName)
    print(temp.info())
    print(temp.columns)

# printType()