import time
import os
import json
from actor import Actor
from anet import ANET_Parameters
from rl import RLSystem
from simworlds.simworld import SimWorld
from visualizer import Visualizer

class TOPP:

    def __init__(self, sim_world: SimWorld, player_count, games_count, total_episodes, search_games, anet_params: ANET_Parameters, train_visualize=False, tournament_visualize=False, frame_delay=0.25, train_epsilon=0.15):
        self.sim_world = sim_world
        self.player_count = player_count
        self.total_episodes = total_episodes
        self.search_games = search_games
        self.games_count = games_count
        self.train_epsilon = train_epsilon
        self.anet_params = anet_params
        self.rl_system = RLSystem(self.sim_world, self.anet_params, epsilon=self.train_epsilon, visualize=train_visualize, frame_delay=frame_delay)
        self.players = []
        self.train_time = 0
        self.visualizer = Visualizer(self.sim_world, frame_delay=frame_delay)
        self.visualizer.set_visualize(tournament_visualize)


    def train_players(self):
        self.train_time = int(time.time())
        train_count = int(self.total_episodes / (self.player_count - 1))
        for i in range(self.player_count):
            player_episodes = i * train_count
            if i:
                self.rl_system.run_episodes(train_count, self.search_games)
            self.rl_system.actor.save_anet(f"./topp/{self.train_time}/{player_episodes}.h5")
        
    def save_params(self, params):
        with open(f"./topp/{self.train_time}/params.json", 'w') as f:
            json.dump(params, f, indent=4)

    def restore_trained_players(self, train_time):
        path = os.walk(f"./topp/{train_time}")
        for root, directories, files in path:
            for file in files:
                if file.endswith(".h5"):
                    new_player = Actor(self.sim_world, self.anet_params)
                    print(root, file)
                    new_player.load_anet(f"{root}/{file}")
                    self.players.append((file, new_player))
        self.train_time = train_time


    def play_tournament(self, save_results=True):
        # Decide if we need to restore players
        if len(self.players) == 0 and self.train_time:
            self.restore_trained_players(self.train_time)
        elif len(self.players) == 0 and not self.train_time:
            return
        
        # Init players scores to 0
        scores = {}
        for p in self.players:
            scores[p[0]] = 0

        reverse_players = sorted(self.players, reverse=True)
        for game_number in range(self.games_count):
            # Alternate between who is going to start the match
            reverse_start = bool(game_number % 2)
            
            # Go through all player, and for each player go through all players who have not played
            for i in range(len(self.players)):
                p1 = reverse_players[i] if reverse_start else self.players[i]
                for j in range(i + 1, len(self.players)):
                    p2 = reverse_players[j] if reverse_start else self.players[j]
                    print(p1[0], p2[0])
                    
                    # Play match between p1 and p2 and add the result
                    result = self.play_actor_match(p1[1], p2[1])
                    if result == 1:
                        scores[p1[0]] += 1
                    elif result == -1:
                        scores[p2[0]] += 1
        
        # Print scores and save results based on arguments
        for player, s in scores.items():
            print(player, s)
        if save_results:
            self.save_scores(scores)

    def play_actor_match(self, p1: Actor, p2: Actor):
        self.sim_world.produce_init_state()
        self.visualizer.init_visualize_episode()
        while not self.sim_world.is_final_state():
            self.visualizer.visualize_state()
            action = self.sim_world.get_action_space()[0]
            if self.sim_world.get_current_player():
                action = p1.get_action(self.sim_world.get_current_encoded_state(), 0)
            else:
                action = p2.get_action(self.sim_world.get_current_encoded_state(), 0)
            print(self.sim_world.get_current_encoded_state(), action)
            self.sim_world.perform_action(action)
        self.visualizer.visualize_final_state()
        result = self.sim_world.get_reward()
        print("Result: ", result)
        print()
        return result

    def save_scores(self, scores):
        # Create file if it doesn't exist
        f = open(f'./topp/{self.train_time}/results.json', 'a+')
        f.close()

        # Read the existing results and append the list and overwrite with the new results
        with open(f'./topp/{self.train_time}/results.json', 'r+') as f:
            score_list = []
            try:
                data = json.load(f)
                score_list.extend(data)
            except json.decoder.JSONDecodeError:
                pass
            score_list.append(scores)
            f.seek(0)
            json.dump(score_list, f, indent=4)
            f.truncate()