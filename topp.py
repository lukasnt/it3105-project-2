import time
import os
import json
from math import floor
from actor import Actor
from learners.anet import ActorNeuralNetwork
from learners.dtrees import DecisionTrees
from learners.learner import Learner
from rl import RLSystem
from simworlds.simworld import SimWorld
from visualizer import Visualizer

class TOPP:

    def __init__(self, sim_world: SimWorld, player_count, games_count, total_episodes, train_search_games, topp_search_games, topp_search_game_delay, learner: Learner, train_visualize=False, tournament_visualize=False, frame_delay=0.25, train_epsilon=0.15):
        self.sim_world = sim_world
        self.player_count = player_count
        self.init_episodes = 0
        self.total_episodes = total_episodes
        self.train_search_games = train_search_games
        self.games_count = games_count
        self.topp_search_games = topp_search_games
        self.topp_search_game_delay = topp_search_game_delay
        self.train_epsilon = train_epsilon
        self.learner = learner
        self.rl_system = RLSystem(self.sim_world, self.learner, epsilon=self.train_epsilon, visualize=train_visualize, frame_delay=frame_delay)
        self.players = []
        self.train_time = 0
        self.visualizer = Visualizer(self.sim_world, frame_delay=frame_delay)
        self.visualizer.set_visualize(tournament_visualize)


    def train_players(self):
        if self.train_time == 0:
            self.train_time = int(time.time())
        train_count = int(self.total_episodes / (self.player_count - 1))
        for i in range(self.player_count):
            player_episodes = self.init_episodes + (i + int(bool(self.init_episodes))) * train_count
            if player_episodes:
                self.rl_system.run_episodes(train_count, self.train_search_games)
            self.rl_system.actor.save_learner(f"./topp/{self.train_time}/{player_episodes}")
            self.rl_system.save_replay_buffer(f"./topp/{self.train_time}")
        
    def save_params(self, params):
        with open(f"./topp/{self.train_time}/params.json", 'w') as f:
            json.dump(params, f, indent=4)

    def restore_trained_players(self, train_time):
        path = os.walk(f"./topp/{train_time}")
        for root, dirs, files in path:
            sorted_dirs = sorted(list(map(lambda d: int(d), dirs)))
            inc = floor(len(sorted_dirs) // (self.player_count - 1))
            if inc == 0:
                inc = 1
            shift = len(sorted_dirs) % inc if inc > 1 else len(sorted_dirs) % (self.player_count - 1)
            i = 0
            while i + shift - 1 < len(sorted_dirs):
                new_index = i + shift - 1 if i else i
                print(new_index)
                dir = str(sorted_dirs[new_index])
                new_player = self.restore_actor_from_dir(root, dir)
                self.players.append((dir, new_player))
                i += inc
            break

        self.train_time = train_time

    def restore_actor_from_dir(self, root, dir):
        new_learner = self.new_learner()
        new_player = Actor(self.sim_world, new_learner, use_mcts=bool(self.topp_search_games) if int(dir) > 0 else False)
        print(root, dir)
        new_player.load_learner(f"{root}/{dir}")
        return new_player

    def restore_rl_trainer(self, train_time, train_opponent=None):
        path = os.walk(f"./topp/{train_time}")
        for root, dirs, files in path:
            if train_opponent:
                self.rl_system.set_opponent(self.restore_actor_from_dir(root, str(train_opponent)))
            max_dir = str(max(map(lambda dir: int(dir), dirs)))
            self.rl_system.actor = self.restore_actor_from_dir(root, max_dir)
            self.init_episodes = int(max_dir)
            break
        self.rl_system.load_replay_buffer(f"./topp/{train_time}")
        self.train_time = train_time

    def new_learner(self) -> Learner:
        new_learner = Learner()
        if isinstance(self.learner, ActorNeuralNetwork):
            new_learner = ActorNeuralNetwork(self.learner.params)
        if isinstance(self.learner, DecisionTrees):
            new_learner = DecisionTrees(self.learner.params)
        return new_learner

    def play_tournament(self, save_results=True):
        # Decide if we need to restore players
        if len(self.players) == 0 and self.train_time:
            self.restore_trained_players(self.train_time)
        elif len(self.players) == 0 and not self.train_time:
            return
        
        # Init players scores to 0
        scores = {}
        for p in self.players:
            scores[p[0]] = (0, 0)

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
                    result = self.play_actor_match(p1[0], p1[1], p2[0], p2[1])
                    p1_score = scores[p1[0]]
                    p2_score = scores[p2[0]]
                    if result == 1:
                        scores[p1[0]] = (p1_score[0] + 1, p1_score[1])
                        scores[p2[0]] = (p2_score[0], p2_score[1] + 1)
                    elif result == -1:
                        scores[p1[0]] = (p1_score[0], p1_score[1] + 1)
                        scores[p2[0]] = (p2_score[0] + 1, p2_score[1])
        
        # Print scores and save results based on arguments
        for player, s in scores.items():
            print(player, s)
        if save_results:
            self.save_scores(scores)

    def play_actor_match(self, p1_name, p1: Actor, p2_name, p2: Actor):
        self.sim_world.produce_init_state()
        self.visualizer.init_visualize_episode(title=f"{p1_name} (BLACK) vs {p2_name} (RED)")
        move_count = 0
        while not self.sim_world.is_final_state():
            self.visualizer.visualize_state()
            t = time.time()
            action = self.sim_world.get_action_space()[0]
            search_games = self.topp_search_games if move_count > self.topp_search_game_delay else 0
            if self.sim_world.get_current_player():
                action = p1.get_action(self.sim_world.get_current_encoded_state(), 0, mcts_episodes=search_games)
            else:
                action = p2.get_action(self.sim_world.get_current_encoded_state(), 0, mcts_episodes=search_games)
            # print(self.sim_world.get_current_encoded_state(), action)
            self.sim_world.perform_action(action)
            # print(action, self.sim_world.get_current_encoded_state())
            d = time.time() - t
            if d > 1:
                print("Time taken: ", d, d < 1)
            move_count += 1
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