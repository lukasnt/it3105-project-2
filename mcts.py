
from math import log, sqrt
import math
import numpy as np
from actor import Actor

from simworlds.simworld import SimWorld
from tree_node import TreeNode


class MonteCarloTreeSearch:

    def __init__(self, sim_world: SimWorld):
        self.sim_world = sim_world
        self.node_visits = {}
        self.edge_visits = {}
        self.tree_policy = {}
        self.root = TreeNode(self.sim_world.get_current_encoded_state(), self.sim_world.get_current_player(), None, None)

    def run_search_game(self, actor: Actor, epsilon):
        self.sim_world.set_current_state(self.root.state, self.root.player)
        leaf_node = self.tree_search(self.root)
        self.node_expansion(leaf_node)
        final_node = self.leaf_evaluation(leaf_node, actor, epsilon)
        self.backpropagation(final_node)

    def move_next_root(self):
        action_space = self.sim_world.get_action_space()
        dist = [self.edge_visits.get((self.root.state, action), 0) / self.node_visits[self.root.state] for action in self.sim_world.get_action_space()]
        best_action = action_space[dist.index(max(dist))]
        print(np.array(self.root.state)[None], np.array(dist)[None])
        print(self.node_visits[self.root.state])
        self.root = next(filter(lambda child : child.prev_action == best_action, self.root.children))
        return self.root, dist

    def get_root(self):
        return self.root

    def exploration_bonus(self, state, to_state, action):
        if self.node_visits.get(to_state, 0) == 0:
            return math.inf
        c = 1
        return c * sqrt(log(self.node_visits.get(to_state, 0)) / (1 + self.edge_visits.get((state, action), 0)))


    def player_action_value(self, state, to_state, action):
        return self.tree_policy.get((state, action), 0) + self.exploration_bonus(state, to_state, action)


    def non_player_action_value(self, state, to_state, action):
        return self.tree_policy.get((state, action), 0) - self.exploration_bonus(state, to_state, action)


    def tree_search(self, root: TreeNode):
        # Stop when finding leaf node
        if len(root.children) == 0:
            return root
        
        # Min-max search
        best_node = root.children[0]
        #for child in root.children:
        #    print(self.tree_policy.get((root.state, child.prev_action), 0))
        if root.player:
            best_node = max(root.children, key=lambda child : self.player_action_value(root.state, child.state, child.prev_action))
        else:
            best_node = min(root.children, key=lambda child : self.non_player_action_value(root.state, child.state, child.prev_action))
        
        # Recurisvly do tree search
        return self.tree_search(best_node)
            

    def node_expansion(self, node: TreeNode):
        self.sim_world.set_current_state(node.state, node.player)
        legal_actions = self.sim_world.get_legal_actions()
        for action in legal_actions:
            self.sim_world.set_current_state(node.state, node.player)
            self.sim_world.perform_action(action)
            state = self.sim_world.get_current_encoded_state()
            player = self.sim_world.get_current_player()
            new_node = TreeNode(state, player, node, action)
            node.add_child(new_node)


    def leaf_evaluation(self, leaf_node: TreeNode, actor: Actor, epsilon = 0):
        node = leaf_node
        self.sim_world.set_current_state(node.state, node.player)

        # Rollout to final state
        while not self.sim_world.is_final_state():
            # Get best action or random action depending on epsilon
            action = actor.get_action(node.state, epsilon)

            # Perform the action and create rollout node
            self.sim_world.perform_action(action)
            new_node = TreeNode(self.sim_world.get_current_encoded_state(), self.sim_world.get_current_player(), node, action, rollout=True)
            node.add_child(new_node)
            node = new_node
        return node


    def backpropagation(self, final_node: TreeNode):
        self.sim_world.set_current_state(final_node.state, final_node.player)
        reward = self.sim_world.get_reward()
        node = final_node
        child_node = None
        while node:
            # Backprop updates
            self.node_visits[node.state] = self.node_visits.get(node.state, 0) + 1
            if child_node:
                prev_eval = self.tree_policy.get((node.state, child_node.prev_action), 0) * (self.node_visits.get(node.state, 0) - 1)
                self.edge_visits[node.state, child_node.prev_action] = self.edge_visits.get((node.state, child_node.prev_action), 0) + 1
                self.tree_policy[node.state, child_node.prev_action] = (prev_eval + reward) / self.node_visits.get(node.state, 0) # (reward - self.tree_policy.get((node.state, child_node.prev_action), 0)) / (self.node_visits.get(node.state, 0) + 1)
            
            # Delete this node if it is a rollout node
            if node.rollout:
                node.parent.children.remove(node)
            
            # Update node pointers for next iteration
            child_node = node
            node = node.parent
