from fileinput import filename
from foldrm import Classifier
import numpy as np
import pandas as pd

def diabetes_012_health_indicators():
    path = "./datasets/"
    fileName = "diabetes_012_health_indicators_BRFSS2015.csv"
    attrs = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']
    nums = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']
    model = Classifier(attrs=attrs, numeric=nums, label='Diabetes_012')
    data = model.load_data(path + fileName)
    print('\n% acute dataset', np.shape(data))
    return model, data


def diabete_binary_5050split():
    path = "./datasets/"
    fileName = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

    attrs = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    nums = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    model = Classifier(attrs=attrs, numeric=nums, label='Diabetes_binary')
    data = model.load_data(path + fileName)
    print('\n% acute dataset', np.shape(data))
    return model, data

def diabetes_binary_health_indicators():
    path = "./datasets/"
    fileName = "diabetes_binary_health_indicators_BRFSS2015.csv"

    attrs = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    nums = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    model = Classifier(attrs=attrs, numeric=nums, label='Diabetes_binary')
    data = model.load_data(path + fileName)
    print('\n% acute dataset', np.shape(data))
    return model, data


def printType():
    path = "./datasets/"
    # fileName = "diabetes_012_health_indicators_BRFSS2015.csv"
    # fileName = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    fileName = "diabetes_binary_health_indicators_BRFSS2015.csv"
    temp = pd.read_csv(path+fileName)
    print(temp.info())
    print(temp.columns)
    # print(list(set(temp['Diabetes_binary'])))

# printType()