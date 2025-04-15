from revivai import SurrogateModel
from palma.base.splitting_strategy import ValidationStrategy
from sklearn.model_selection import ShuffleSplit
from utils.project import Project
from utils.engine import FlamlOptimizer


class SurrogateModeling:
    def __init__(self, estimator_list, problem, project_name="default"):
        self.__model = None
        self.estimator_list = estimator_list
        self.problem = problem
        self.project_name = project_name

    def get_best_model(self, X, y, add_learner=False, learner=None):
        """Function that aims to select the best model, the user can also add learner to flaml
        :param X: data for training
        :param y: data for training
        :param add_learner bool, True to add a new learner False if else
        :param learner dict, with new learner to add
        """
        engine_parameters = {
            "time_budget": 30,
            "metric": "r2",
            "log_training_metric": True,
            "estimator_list": self.estimator_list
        }
        splitting_strategy = ValidationStrategy(splitter=ShuffleSplit(
            n_splits=10, random_state=1,
        ))
        X, y = splitting_strategy(X, y)
        self.X = X
        self.y = y

        # Project creation
        project = Project(problem=self.problem, project_name=self.project_name)
        project.start(X, y, splitter=ShuffleSplit(n_splits=10, random_state=42), )
        # Create and start optimizer
        if add_learner:
            optimizer = FlamlOptimizer(engine_parameters, learner)
        else:
            optimizer = FlamlOptimizer(engine_parameters, {})
        optimizer.start(project)

        # Get the best model
        best_model = optimizer.best_model_
        self.__model = best_model

    @property
    def model(self):
        """Function that return the best model selected"""
        return self.__model

    def save(self, folder_name, file_name):
        """Function aims to save the model, the data and prediction in hdf5 file
        :param folder_name: The folder name where to save the hfd5 file
        :param file_name: The file name where to save the model, the data and the prediction
        """
        srgt_model = SurrogateModel()
        srgt_model.set(self.X, self.y, self.model)
        srgt_model.dump(folder_name, file_name)
