from foldrm import Classifier
import numpy as np
import pandas as pd

def diabete():
    attrs = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    # nums = ["Diabetes_binary"]
    nums = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
        "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
    model = Classifier(attrs=attrs, numeric=nums, label='Diabetes_binary')
    data = model.load_data('diabete/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    print('\n% acute dataset', np.shape(data))
    return model, data


def printType():
    temp = pd.read_csv('diabete/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    print(temp.info())
    # print(list(set(temp['Diabetes_binary'])))