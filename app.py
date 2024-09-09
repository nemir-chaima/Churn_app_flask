# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import statsmodels.api as sm
import pickle
import numpy as np

app = Flask(__name__)

with open('logistic_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    final_features = sm.add_constant(final_features, has_constant='add')
 
    prediction = loaded_model.predict(final_features)
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Churn Probability: {:.2f}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
