
class Learner:

    def __init__(self):
        self.params = None
    
    def init_model(self): pass
    def train_model(self, replay_buffer): pass
    def get_dist(self, state): pass
    def save_model_to_file(self, filepath): pass
    def load_model_from_file(self, filepath): pass