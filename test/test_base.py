import pandas as pd
import os
from sklearn.datasets import make_classification
from base.surrogate_model import SurrogateModeling
import numpy as np
from utils.NN_model import KerasNN
from utils.kriging_model import KRGModel


def test_save():
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Optimizer Flaml parameters
    estimator_list = ['catboost', 'xgboost', "lgbm", "KRG", 'keras']
    learners = {"KRG": KRGModel, 'keras': KerasNN}
    # Instanciate the metamodel
    srgt = SurrogateModeling(estimator_list=estimator_list, problem='classification', project_name='default')
    # Get the best model, wihth adding new learner to flaml
    srgt.get_best_model(X, y, add_learner=True, learner=learners)
    # Save the model
    srgt.save("./metamodel_test", 'file_test')
    # Asserts
    assert np.allclose(srgt.X, X), "The data X are not matching"
    assert np.allclose(srgt.y, y), "The data y are not matching"
    assert os.path.exists('./metamodel_test/file_test')