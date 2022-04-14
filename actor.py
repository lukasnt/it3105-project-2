
import random
from learners.learner import Learner
from simworlds.simworld import SimWorld

class Actor:

    def __init__(self, sim_world: SimWorld, learner: Learner):
        self.sim_world = sim_world
        self.learner = learner

    def get_action(self, state, epsilon):
        dist = self.learner.get_dist(state)

        # Set all actions that are illegal to 0
        legal_actions = self.sim_world.get_legal_actions()
        action_space = self.sim_world.get_action_space()
        for i in range(len(action_space)):
            action = action_space[i]
            if action not in legal_actions:
                dist[i] = 0

        # Find best action based on ANET distribution output
        best_action = self.sim_world.get_action_space()[dist.index(max(dist))]
        # print(state, dist, best_action)

        # Use best action or random action based on epsilon value
        action = best_action
        if random.random() < epsilon or action not in legal_actions:
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
        return action

    def init_learner(self):
        self.learner.init_model()

    def train_learner(self, replay_buffer):
        self.learner.train_model(replay_buffer)

    def load_learner(self, filepath):
        self.learner.load_model_from_file(filepath)
    
    def save_learner(self, filepath):
        self.learner.save_model_to_file(filepath)
