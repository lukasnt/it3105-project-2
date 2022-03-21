
from math import log, sqrt
import random
import tensorflow as tf
import numpy as np

from simworlds.simworld import SimWorld
from tree_node import TreeNode


class MonteCarloTreeSearch:

    save_interval = 10
    replay_buffer = []

    node_visits = {}
    edge_visits = {}
    tree_policy = {}

    epsilon = 0.5
    epsilon_decay = 0.1

    def __init__(self, sim_world: SimWorld, save_interval = 10):
        self.sim_world = sim_world
        self.save_interval = save_interval
        self.init_actor_network()
    

    def init_actor_network(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.sim_world.get_encoding_shape()),
            tf.keras.layers.Dense(16, activation="softmax"),
            tf.keras.layers.Dense(16, activation="softmax"),
            tf.keras.layers.Dense(len(self.sim_world.get_action_space()))
        ])

        self.model.compile(
            optimizer=tf.optimizers.SGD(learning_rate=0.01),
            loss=tf.losses.MeanSquaredError(),
            metrics=["accuracy"]
        )


    def run_episodes(self, episodes, search_games):
        self.epsilon_decay = self.epsilon / (episodes - 1)
        for episode_num in range(episodes):
            self.reset_episode_data()
            self.sim_world.produce_init_state()
            root = TreeNode(self.sim_world.get_current_state(), self.sim_world.get_current_player(), None, None)
            while not self.sim_world.is_final_state():
                self.sim_world.set_current_state(root.state, root.player)
                for search_game_num in range(search_games):
                    leaf_node = self.tree_search(root)
                    self.node_expansion(leaf_node)
                    final_node = self.leaf_evaluation(leaf_node, self.epsilon)
                    self.backpropagation(final_node)
                action_space = self.sim_world.get_action_space()
                dist = [self.edge_visits.get((root.state, action), 0) / self.node_visits[root.state] for action in self.sim_world.get_action_space()]
                self.replay_buffer.append((root.state, dist))
                best_action = action_space[dist.index(max(dist))]
                print(np.array(root.state)[None], np.array(dist)[None])
                print(self.node_visits[root.state])
                root = next(filter(lambda child : child.prev_action == best_action, root.children))
                self.sim_world.set_current_state(root.state, root.player)
            self.train_anet()
            self.epsilon -= self.epsilon_decay


    def reset_episode_data(self):
        self.replay_buffer = []
        self.node_visits = {}
        self.edge_visits = {}
        self.tree_policy = {}

    def exploration_bonus(self, state, action):
        c = 1
        return c * sqrt(log(1 + self.node_visits.get(state, 0)) / (1 + self.edge_visits.get((state, action), 0)))


    def action_value(self, state, action):
        return self.tree_policy.get((state, action), 0) + self.exploration_bonus(state, action)


    def train_anet(self):
        states = np.array(list(map(lambda b: b[0], self.replay_buffer)))
        dists = np.array(list(map(lambda b: b[1], self.replay_buffer)))
        for i in range(len(states)):
            self.model.fit(states[i][None], dists[i][None])


    def tree_search(self, root: TreeNode):
        # Stop when finding leaf node
        if len(root.children) == 0:
            return root
        
        # print(root)
        # print(root.children)

        # Min-max search
        best_node = root.children[0]
        if root.player:
            best_node = max(root.children, key=lambda child : self.tree_policy.get((child.state, child.prev_action), 0) + self.exploration_bonus(child.state, child.prev_action))
        else:
            best_node = min(root.children, key=lambda child : self.tree_policy.get((child.state, child.prev_action), 0) - self.exploration_bonus(child.state, child.prev_action))
        
        # Recurisvly do tree search
        return self.tree_search(best_node)
            

    def node_expansion(self, node: TreeNode):
        # print("root_node: ", node)
        for action in self.sim_world.get_legal_actions():
            #print("Player object:", node.player)
            self.sim_world.set_current_state(node.state, node.player)
            #print("Player before: ", self.sim_world.get_current_player())
            self.sim_world.perform_action(action)
            state = self.sim_world.get_current_state()
            player = self.sim_world.get_current_player()
            #print(state, player)
            new_node = TreeNode(state, player, node, action)
            # print("new_node:", new_node)
            node.add_child(new_node)


    def leaf_evaluation(self, leaf_node: TreeNode, epsilon = 0):
        action_space = self.sim_world.get_action_space()
        node = leaf_node
        self.sim_world.set_current_state(node.state, node.player)

        # Rollout to final state
        while not self.sim_world.is_final_state():
            dist = self.model(np.array(node.state)[None]).numpy().tolist()[0]
            # print(dist)

            # Find best action based on ANET distribution output
            best_action = action_space[dist.index(max(dist))]

            # Use best action or random action based on epsilon value
            action = best_action
            legal_actions = self.sim_world.get_legal_actions()
            if random.random() < epsilon or action not in legal_actions:
                action = legal_actions[random.randint(0, len(legal_actions) - 1)]

            # Perform the action and create rollout node
            self.sim_world.perform_action(action)
            new_node = TreeNode(self.sim_world.get_current_state(), self.sim_world.get_current_player(), node, action, rollout=True)
            node.add_child(new_node)
            # print("rollout_nodes: ", node, new_node)
            # print(new_node.state)
            node = new_node
        return node


    def backpropagation(self, final_node: TreeNode):
        self.sim_world.set_current_state(final_node.state, final_node.player)
        reward = self.sim_world.get_reward()
        # print("reward:", reward)
        node = final_node
        child_node = None
        # print("backprop_node:", node)
        # print("backprop_node_children: ", node.children)
        while node:
            # Backprop updates
            self.node_visits[node.state] = self.node_visits.get(node.state, 0) + 1
            if child_node:
                prev_eval = self.tree_policy.get((node.state, child_node.prev_action), 0) * (self.node_visits.get(node.state, 0) - 1)
                # print("prev_eval:", prev_eval)
                self.edge_visits[node.state, child_node.prev_action] = self.edge_visits.get((node.state, child_node.prev_action), 0) + 1
                self.tree_policy[node.state, child_node.prev_action] = (prev_eval + reward) / self.node_visits.get(node.state, 0)
            
            # print("children before:", node.children)
            # Delete this node if it is a rollout node
            if node.rollout:
                node.parent.children.remove(node)
            # print("children after:", node.children)
            
            # Update node pointers for next iteration
            child_node = node
            node = node.parent
