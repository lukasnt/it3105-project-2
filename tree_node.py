
class TreeNode:

    state = None
    player = None
    parent = None
    rollout = False
    prev_action = None
    children = []

    def __init__(self, state, player, parent, prev_action, rollout = False):
        self.state = state
        self.player = player
        self.parent = parent
        self.rollout = rollout
        self.prev_action = prev_action
        self.children = []

    def add_child(self, tree_node):
        self.children.append(tree_node)

    def set_parent(self, parent_node):
        self.parent = parent_node
