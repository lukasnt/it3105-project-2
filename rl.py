
from actor import Actor
from learners.learner import Learner
from mcts import MonteCarloTreeSearch
from simworlds.simworld import SimWorld
from visualizer import Visualizer

class RLSystem:

    def __init__(self, sim_world: SimWorld, learner: Learner, epsilon=0.15, save_interval = 10, visualize=False, frame_delay=0.25):
        self.sim_world = sim_world
        self.save_interval = save_interval
        self.actor = Actor(self.sim_world, learner)
        self.actor.init_learner()
        self.epsilon = epsilon
        self.visualizer = Visualizer(self.sim_world, frame_delay=frame_delay)
        self.visualizer.set_visualize(visualize)
        self.replay_buffer = []
        self.episode_buffer = []
        self.save_interval = 10

    def run_episodes(self, episodes, search_games):
        self.epsilon_decay = self.epsilon
        if episodes > 1:
            self.epsilon_decay = self.epsilon / (episodes - 1)
        for episode_num in range(episodes):
            self.reset_episode_data()
            self.sim_world.produce_init_state()
            mcts = MonteCarloTreeSearch(self.sim_world)
            root = mcts.get_root()
            self.visualizer.init_visualize_episode()
            while not self.sim_world.is_final_state():
                self.visualizer.visualize_state()
                for search_game_num in range(search_games):
                    mcts.run_search_game(self.actor, self.epsilon)
                prev_root = root
                root, dist = mcts.move_next_root()
                self.episode_buffer.append((prev_root.state, dist))
                self.sim_world.set_current_state(root.state, root.player)
            self.visualizer.visualize_final_state()
            self.update_buffers(1 if self.sim_world.get_reward() == 1 else 0)
            self.actor.train_learner(self.replay_buffer)
            self.epsilon -= self.epsilon_decay

    def update_buffers(self, winner):
        self.episode_buffer = list(filter(lambda b: b[0][len(b[0]) - 1] == winner, self.episode_buffer))
        self.replay_buffer.extend(self.episode_buffer)
        self.reset_episode_data() 

    def reset_episode_data(self):
        self.episode_buffer = []