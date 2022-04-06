class SimWorld:
    
    def produce_init_state(self): pass
    def get_action_space(self): pass
    def get_legal_actions(self): pass
    def perform_action(self, action): pass
    def is_final_state(self): pass
    def get_reward(self): pass
    def get_encoding_shape(self): pass
    def get_current_encoded_state(self): pass
    def get_current_player(self): pass
    def set_current_state(self, encoded_state, player): pass
    def visualize_state(self, ax): pass
