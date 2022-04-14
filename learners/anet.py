import tensorflow as tf
import numpy as np
from keras.models import load_model

from learners.learner import Learner

class ANET_Parameters(Learner):
    
    def __init__(self, input_shape, action_space, dimensions, learning_rate, activation="softmax", optimizer="SGD"):
        self.dimensions = dimensions
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer

class ActorNeuralNetwork:

    def __init__(self, params: ANET_Parameters):
        self.params = params
        self.model = None

    def init_model(self):
        layers = []
        layers.append(tf.keras.layers.Input(shape=self.params.input_shape))
        for dimension in self.params.dimensions:
            layers.append(tf.keras.layers.Dense(dimension, activation=self.params.activation))
        layers.append(tf.keras.layers.Dense(len(self.params.action_space), activation=self.params.activation))   
        self.model = tf.keras.Sequential(layers)

        net_optimizer = None
        if self.params.optimizer == "SGD":
            net_optimizer = tf.optimizers.SGD(learning_rate=self.params.learning_rate)
        elif self.params.optimizer == "Adagrad":
            net_optimizer = tf.optimizers.Adagrad(learning_rate=self.params.learning_rate)
        elif self.params.optimizer == "RMSProp":
            net_optimizer = tf.optimizers.RMSprop(learning_rate=self.params.learning_rate)
        elif self.params.optimizer == "Adam":
            net_optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate)

        self.model.compile(
            optimizer=net_optimizer,
            loss=tf.losses.MeanSquaredError(),
            metrics=["accuracy"]
        )
        self.model.summary()

    def train_model(self, replay_buffer):
        states = np.array(list(map(lambda b: b[0], replay_buffer)))
        dists = np.array(list(map(lambda b: b[1], replay_buffer)))
        for i in range(len(states)):
            print(states[i][None], dists[i][None])
            self.model.fit(states[i][None], dists[i][None])
    
    def get_dist(self, state):
        # print("get_dist:", np.array(state)[None])
        return self.model(np.array(state)[None]).numpy().tolist()[0]

    def save_model_to_file(self, filepath):
        self.model.save(f"{filepath}/model.h5")
    
    def load_model_from_file(self, filepath):
        self.model = load_model(f"{filepath}/model.h5")
