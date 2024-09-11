import unittest
import pickle
import pandas as pd
import numpy as np


def load_model():
    with open('random_forest_t.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def generate_input():
    age = np.random.randint(18, 100)
    years = np.random.randint(0, 30)
    num_sites = np.random.randint(0, 50)  
    input_data = pd.DataFrame([[age, years, num_sites]])
    return input_data

def test_prediction_case_1(model):
    try:
        input_data = generate_input()
        prediction = model.predict(input_data)
        return 'no error' 
    except Exception as e:
        return(f"Error during prediction: {e}")
         

def test_process():
     model = load_model()
     print('model load fait')
     print(test_prediction_case_1(model))
test_process()
