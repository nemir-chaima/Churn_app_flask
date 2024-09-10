import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , StandardScaler , RobustScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data():
    data = pd.read_csv('customer_churn.csv')
    return data

def data_split(data):
    columns_to_keep = ['Age', 'Years' ,'Num_Sites', 'Churn']
    data = data[columns_to_keep]
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train , X_test, y_train, y_test

def normalisation(X_train , X_test):
    MN = MinMaxScaler()
    x_train_scaled = MN.fit_transform(X_train)
    x_test_scaled = MN.fit_transform(X_test)
    return x_train_scaled, x_test_scaled

def train_model(x_train_scaled, y_train):
    rf = RandomForestClassifier()
    rf.fit(x_train_scaled, y_train)
    return rf

def train_process():
    data = load_data()
    print('load fait')
    X_train , X_test, y_train, y_test = data_split(data)
    print('split fait')
    x_train_scaled, x_test_scaled = normalisation(X_train , X_test)
    print('normalisation fait')
    rf = train_model(x_train_scaled, y_train)
    print('model fait')
    import pickle
    with open('random_forest_t.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print('save fait ')

train_process()