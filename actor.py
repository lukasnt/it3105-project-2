
import random
from anet import ANET_Parameters, ActorNeuralNetwork
from simworlds.simworld import SimWorld

class Actor:

    def __init__(self, sim_world: SimWorld, anet_params: ANET_Parameters):
        self.sim_world = sim_world
        self.anet = ActorNeuralNetwork(anet_params)

    def get_action(self, state, epsilon):
        dist = self.anet.get_dist(state)

        # Find best action based on ANET distribution output
        best_action = self.sim_world.get_action_space()[dist.index(max(dist))]
        # print(state, dist, best_action)

        # Use best action or random action based on epsilon value
        action = best_action
        legal_actions = self.sim_world.get_legal_actions()
        if random.random() < epsilon or action not in legal_actions:
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
        return action

    def init_anet(self):
        self.anet.init_anet()

    def train_anet(self, replay_buffer):
        self.anet.train_anet(replay_buffer)

    def load_anet(self, filepath):
        self.anet.load_anet_from_file(filepath)
    
    def save_anet(self, filepath):
        self.anet.save_anet_to_file(filepath)
