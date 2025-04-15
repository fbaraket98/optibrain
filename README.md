
# OptiBrain

A python package that aims to select automatically the best model for your tasks and save the trained model..

## Install

`pip install optibrain@git+`

## Simple usage

* Auto-select and save model.

```python
import pandas as pd

from sklearn.datasets import make_classification
from base.surrogate_model import SurrogateModeling
from utils.kriging_model import KRGModel
import numpy as np

X, y = make_classification()
X = pd.DataFrame(X)
y = pd.Series(y)
estimator_list = ["catboost", 'xgboost', 'lgbm', 'KRG', 'keras']
# instanciate the metamodel
srgt = SurrogateModeling(estimator_list=estimator_list, problem='classification')
# select the best model
srgt.get_best_model(X, y, add_learner=True, learner={"KRG": KRGModel})

# save the model
srgt.save("./metamodel_folder", "file_name")
```

In the method get_best_model, the user can add new learner, by setting True to add_learner and adding learner dictionary with
the names of the learner and their classes.  
The result of this example is a HDF5 file where the information about the selected model are saved.


