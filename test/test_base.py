import pandas as pd
import os
from sklearn.datasets import make_classification
from base.surrogate_model import SurrogateModeling
import numpy as np
from utils.NN_model import FullNeuralNetwork
from utils.kriging_model import KRGModel


def test_save():
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Optimizer Flaml parameters
    estimator_list = ['catboost', 'xgboost', "lgbm", "KRG", 'RN']
    learners = {"KRG": KRGModel, 'RN': FullNeuralNetwork}
    # Instanciate the metamodel
    srgt = SurrogateModeling(estimator_list=estimator_list, problem='regression', project_name='default')
    # Get the best model, wihth adding new learner to flaml
    srgt.get_best_model(X, y, add_learner=True, learner=learners)
    # Save the model
    # Asserts
    assert np.allclose(srgt.X, X), "The data X are not matching"
    assert np.allclose(srgt.y, y), "The data y are not matching"


def test_multioutput():
    import pickle
    from sklearn.model_selection import train_test_split
    path = "C:/Users/Fatma/Downloads/WireModelData/DB_wiremodel24.pickle"
    with open(path, "rb") as f:
        x = pickle.load(f)
    X = pd.DataFrame(x['INPUT']['data'])
    X.columns = ['tension', 'amplitude', 'deplacement_serrage',
                 'deplacement_poids_propre', 'span_length', 'denivele']
    y = x['OUTPUT']['data']

    # print(len(y))
    # print(len(y[0]))
    # exit()

    outputs = pd.DataFrame(y)
    outputs.columns = ["min", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "max"]
    list_target = ['min', '40%', '10%', '90%']
    list_features = ['tension', 'amplitude', "deplacement_poids_propre"]
    y_selected = outputs[list_target]
    X = X[list_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y_selected, test_size=.2)
    srgm = SurrogateModeling(['catboost','RN','lgbm','xgboost','KRG'],'regression')
    learners = {"RN":FullNeuralNetwork,'KRG':KRGModel}
    srgm.get_best_model(X_train, y_train, add_learner=True, learner=learners)
    assert np.allclose(srgm.X, X_train), "The data X are not matching"
    assert np.allclose(srgm.y,y_train), "The data y are not matching"

