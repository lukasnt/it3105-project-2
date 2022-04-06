from matplotlib.patches import Rectangle
from simworlds.simworld import SimWorld


class Nim(SimWorld):

    def __init__(self, start_pieces, max_move, start_player):
        self.start_pieces = start_pieces
        self.pieces = start_pieces
        self.max_move = max_move
        self.start_player_turn = start_player
        self.player_turn = start_player

    def produce_init_state(self):
        self.pieces = self.start_pieces
        self.player_turn = self.start_player_turn

    def get_action_space(self):
        return [i for i in range(1, self.max_move + 1)]

    def get_legal_actions(self):
        # print(self.max_move, self.pieces)
        return [action for action in range(1, min(self.max_move, self.pieces) + 1)]
    
    def perform_action(self, action):
        self.pieces -= action
        self.player_turn = not self.player_turn
    
    def is_final_state(self):
        return self.pieces <= 0
    
    def get_reward(self):
        if self.is_final_state():
            return -1 if self.player_turn else 1
        else:
            return 0
        # return -1 if self.player_turn else 1 if self.is_final_state() else 0
    
    def get_current_encoded_state(self):
        return (self.pieces, self.player_turn)
    
    def get_encoding_shape(self):
        return (2, )

    def get_current_player(self):
        return self.player_turn

    def set_current_state(self, encoded_state, player):
        self.pieces = encoded_state[0]
        self.player_turn = player
    
    def set_current_player(self, player_turn):
        self.player_turn = player_turn

    def visualize_state(self, ax):
        ax.set(ylim=[0, 1], xlim=[0, self.start_pieces + 1])
        for i in range(1, self.pieces + 1):
            rect = Rectangle((i, 0.25), 0.8, 0.5, color="black")
            ax.add_patch(rect)