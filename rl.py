
from actor import Actor
from anet import ANET_Parameters
from mcts import MonteCarloTreeSearch
from simworlds.simworld import SimWorld
from visualizer import Visualizer

class RLSystem:

    save_interval = 10
    replay_buffer = []

    def __init__(self, sim_world: SimWorld, anet_params: ANET_Parameters, save_interval = 10, visualize=False, frame_delay=0.25):
        self.sim_world = sim_world
        self.save_interval = save_interval
        self.actor = Actor(self.sim_world, anet_params)
        self.actor.anet.init_anet()
        self.epsilon = 0.15
        self.visualizer = Visualizer(self.sim_world, frame_delay=frame_delay)
        self.visualizer.set_visualize(visualize)


    def run_episodes(self, episodes, search_games):
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
                self.replay_buffer.append((prev_root.state, dist))
                self.sim_world.set_current_state(root.state, root.player)
            self.visualizer.visualize_final_state()
            self.filter_replay_buffer(1 if self.sim_world.get_reward() == 1 else 0)
            self.actor.anet.train_anet(self.replay_buffer)
            self.epsilon -= self.epsilon_decay

    def filter_replay_buffer(self, winner):
        self.replay_buffer = list(filter(lambda b: b[0][len(b[0]) - 1] == winner, self.replay_buffer))
        print(self.replay_buffer)   

    def reset_episode_data(self):
        self.replay_buffer = []