
import os
import numpy as np
from learners.learner import Learner
from sklearn.tree import DecisionTreeRegressor
import sklearn
import random
import pickle

class DecisionTreesParams:

    def __init__(self, tree_count):
        self.tree_count = tree_count

class DecisionTrees(Learner):

    def __init__(self, params : DecisionTreesParams):
        self.params = params
        self.tree_count = params.tree_count
        self.models = []

    def init_model(self):
        self.models = [DecisionTreeRegressor(random_state=0) for _ in range(self.tree_count)]
    
    def train_model(self, replay_buffer):
        states = np.array(list(map(lambda b: b[0], replay_buffer)))
        dists = np.array(list(map(lambda b: b[1], replay_buffer)))
        for i in range(self.tree_count):
            labels = list(map(lambda d: d[i], dists))
            # print(states, labels)
            self.models[i].fit(states, labels)

    def get_dist(self, state):
        dist = []
        for model in self.models:
            try:
                dist.append(model.predict([state])[0])
            except sklearn.exceptions.NotFittedError:
                dist.append(random.random())
        # print(state, dist)
        return dist

    def save_model_to_file(self, filepath):
        fullpath = os.getcwd() + filepath[1:].replace("/", os.sep)
        os.makedirs(fullpath, exist_ok=True)
        for i in range(self.tree_count):
            with open(f"{fullpath}{os.sep}{i}.pickle", "wb") as f:
                pickle.dump(self.models[i], f)

    def load_model_from_file(self, filepath):
        self.models = []
        fullpath = os.getcwd() + filepath[1:].replace("/", os.sep)
        for i in range(self.tree_count):
            with open(f"{fullpath}{os.sep}{i}.pickle", "rb") as f:
                self.models.append(pickle.load(f))