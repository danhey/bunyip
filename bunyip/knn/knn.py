from sklearn import neighbors
import pickle
import os

__all__ = ["KNN"]

class KNN(object):
    def __init__(self, model_path=None):
        # load the model from disk
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__),  'knnpickle_file')
        self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, x):
        return self.model.predict(x[None,:])

    def retrain(self):
        pass