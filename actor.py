
import random
from learners.learner import Learner
from mcts import MonteCarloTreeSearch
from simworlds.simworld import SimWorld

class Actor:

    def __init__(self, sim_world: SimWorld, learner: Learner, use_mcts=False):
        self.sim_world = sim_world
        self.learner = learner
        self.use_mcts = use_mcts
        if self.use_mcts:
            self.mcts = MonteCarloTreeSearch(self.sim_world)

    def get_action(self, state, epsilon, mcts_episodes=0, check_reward=True):
        if check_reward:
            action = self.check_winning(state)
            # print("Winning action", action)
            if action:
                return action
            action = self.check_losing(state)
            # print("Losing Action", action)
            if action:
                return action
            action = self.check_winning_fork(state)
            if action:
                # print("Winning fork", action)
                return action
            action = self.check_losing_fork(state)
            if action:
                # print("Losing fork", action)
                return action
            """
            action = self.check_winning_quad_fork(state)
            if action:
                print("Winning Quad fork", action)
                return action
            action = self.check_loosing_quad_fork(state)
            if action:
                print("Losing Quad fork", action)
                return action
            """

        # Get dist from learner or dist from mcts run with itself as actor
        if mcts_episodes and self.use_mcts:
            player = self.sim_world.get_current_player()
            self.mcts.manual_set_root(state)
            for _ in range(mcts_episodes):
                self.mcts.run_search_game(self, epsilon)
            new_root, dist = self.mcts.move_next_root()
            self.sim_world.set_current_state(state, player)
        else:
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

    def check_winning(self, state):
        player = self.sim_world.get_current_player()
        legal_actions = self.sim_world.get_legal_actions()
        found_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            reward = self.sim_world.get_reward()
            if reward == 1 and player or reward == -1 and not player:
                found_action = action
            self.sim_world.set_current_state(state, player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_action

    def check_losing(self, state):
        player = self.sim_world.get_current_player()
        opposite_player = not player
        self.sim_world.set_current_state(self.sim_world.get_current_encoded_state(), opposite_player)
        legal_actions = self.sim_world.get_legal_actions()
        found_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            reward = self.sim_world.get_reward()
            if reward == 1 and opposite_player or reward == -1 and not opposite_player:
                found_action = action
            self.sim_world.set_current_state(state, opposite_player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_action

    def check_winning_fork(self, state):
        player = self.sim_world.get_current_player()
        legal_actions = self.sim_world.get_legal_actions()
        found_fork_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            temp_state = self.sim_world.get_current_encoded_state()
            won_count = 0
            for n_action in self.sim_world.bfs_tree_neighbors((action[0], action[1]), (1, 0) if player else (0, 1)):
                self.sim_world.set_current_state(temp_state, player)
                if n_action in legal_actions:   
                    self.sim_world.perform_action(n_action)
                    reward = self.sim_world.get_reward()
                    if reward == 1 and player or reward == -1 and not player:
                        won_count += 1
            if won_count >= 2:
                found_fork_action = action
            self.sim_world.set_current_state(state, player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_fork_action

    
    def check_losing_fork(self, state):
        player = self.sim_world.get_current_player()
        opposite_player = not player
        self.sim_world.set_current_state(self.sim_world.get_current_encoded_state(), opposite_player)
        legal_actions = self.sim_world.get_legal_actions()
        found_fork_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            temp_state = self.sim_world.get_current_encoded_state()
            lose_count = 0
            for n_action in self.sim_world.bfs_tree_neighbors((action[0], action[1]), (1, 0) if opposite_player else (0, 1)):
                self.sim_world.set_current_state(temp_state, opposite_player)
                if n_action in legal_actions:   
                    self.sim_world.perform_action(n_action)
                    reward = self.sim_world.get_reward()
                    if reward == 1 and opposite_player or reward == -1 and not opposite_player:
                        lose_count += 1
            if lose_count >= 2:
                found_fork_action = action
            self.sim_world.set_current_state(state, opposite_player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_fork_action

    def check_winning_quad_fork(self, state):
        player = self.sim_world.get_current_player()
        legal_actions = self.sim_world.get_legal_actions()
        found_quad_fork_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            temp_state = self.sim_world.get_current_encoded_state()
            fork_count = 0
            for n_action in self.sim_world.bfs_tree_neighbors((action[0], action[1]), (1, 0) if player else (0, 1)):
                self.sim_world.set_current_state(temp_state, player)
                if n_action in legal_actions:   
                    self.sim_world.perform_action(n_action)
                    fork_action = self.check_winning_fork(self.sim_world.get_current_encoded_state())
                    if fork_action:
                        fork_count += 1
            if fork_count >= 2:
                found_quad_fork_action = action
            self.sim_world.set_current_state(state, player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_quad_fork_action

    def check_loosing_quad_fork(self, state):
        player = self.sim_world.get_current_player()
        opposite_player = not player
        self.sim_world.set_current_state(self.sim_world.get_current_encoded_state(), opposite_player)
        legal_actions = self.sim_world.get_legal_actions()
        found_quad_fork_action = None
        for action in legal_actions:
            self.sim_world.perform_action(action)
            temp_state = self.sim_world.get_current_encoded_state()
            fork_count = 0
            for n_action in self.sim_world.bfs_tree_neighbors((action[0], action[1]), (1, 0) if opposite_player else (0, 1)):
                self.sim_world.set_current_state(temp_state, opposite_player)
                if n_action in legal_actions:   
                    self.sim_world.perform_action(n_action)
                    fork_action = self.check_winning_fork(self.sim_world.get_current_encoded_state())
                    if fork_action:
                        fork_count += 1
            if fork_count >= 2:
                found_quad_fork_action = action
            self.sim_world.set_current_state(state, opposite_player)
        # Reset back to original state
        self.sim_world.set_current_state(state, player)
        return found_quad_fork_action

    def init_learner(self):
        self.learner.init_model()

    def train_learner(self, replay_buffer):
        self.learner.train_model(replay_buffer)

    def load_learner(self, filepath):
        self.learner.load_model_from_file(filepath)
    
    def save_learner(self, filepath):
        self.learner.save_model_to_file(filepath)
