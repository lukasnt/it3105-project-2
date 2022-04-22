# Import and initialize your own actor
from actor import Actor
from visualizer import Visualizer

# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
class MyClient(ActorClient):

    def __init__(self, actor: Actor, visualize=True, frame_delay=0.01, search_games=0, search_games_delay=0, auth="", qualify=False):
        super().__init__(auth=auth, qualify=qualify)
        self.actor = actor
        self.is_start_player = False
        self.start_player_id = 1
        self.my_series_id = 1
        self.search_games = search_games
        self.search_games_delay = search_games_delay
        self.move_count = 0
        self.visualizer = Visualizer(self.actor.sim_world, frame_delay=frame_delay)
        self.visualizer.set_visualize(visualize)

    def handle_get_action(self, state):
        """Called whenever it's your turn to pick an action

        Args:
            state (list): board configuration as a list of board_size^2 + 1 ints

        Returns:
            tuple: action with board coordinates (row, col) (a list is ok too)

        Note:
            > Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
            > Player 1 goes "top-down" and player 2 goes "left-right"
            > Returning (3, 2) would put a "1" at the free (0) position
              below the two vertically aligned ones.
            > The neighborhood around a cell is connected like
                  |/
                --0--
                 /|
        """
        self.move_count += 1
        # print(state, self.state_to_state_encoding(state))
        encoded_state = self.state_to_state_encoding(state)
        flipped_encoded_state = self.encoded_board_flipped(encoded_state)
        self.actor.sim_world.set_current_state(flipped_encoded_state if self.my_series_id == 1 else encoded_state, True)
        self.visualizer.visualize_state()
        search_games = self.search_games if self.move_count > self.search_games_delay else 0
        check_reward = self.move_count >= 7 and self.move_count <= self.search_games_delay
        epsilon = 1 if self.move_count <= 4 else 0
        row, col = self.actor.get_action(self.actor.sim_world.get_current_encoded_state(), epsilon, mcts_episodes=search_games, check_reward=check_reward) # Your logic
        self.move_count += 1
        # Flip col and row if we are red (id == 1) externaly
        result = (col, row) if self.my_series_id == 1 else (row, col)
        print(result)
        return result

    def handle_game_start(self, start_player):
        """Called at the beginning of of each game
        Args:
        start_player (int): the series_id of the starting player (1 or 2)
        """
        self.move_count = 0
        self.start_player_id = start_player
        self.is_start_player = True if start_player == self.my_series_id else False
        self.actor.sim_world.set_start_player(self.is_start_player)
        self.actor.sim_world.produce_init_state()
        color = "BLACK" # "RED" if self.my_series_id == 1 else "BLACK"
        self.visualizer.init_visualize_episode(title=f"We are {color}")
        self.visualizer.visualize_state()
        self.logger.info('Game start: start_player=%s', start_player)
    
    def handle_game_over(self, winner, end_state):
        """Called after each game

        Args:
            winner (int): the winning player (1 or 2)
            end_stats (tuple): final board configuration

        Note:
            > Given the following end state for a 5x5 Hex game
            state = [
                2,              # Current player is 2 (doesn't matter)
                0, 2, 0, 1, 2,  # First row
                0, 2, 1, 0, 0,  # Second row
                0, 0, 1, 0, 0,  # ...
                2, 2, 1, 0, 0,
                0, 1, 0, 0, 0
            ]
            > Player 1 has won here since there is a continuous
              path of ones from the top to the bottom following the
              neighborhood description given in `handle_get_action`
        """
        encoded_state = self.state_to_state_encoding(end_state)
        flipped_encoded_state = self.encoded_board_flipped(encoded_state)
        self.actor.sim_world.set_current_state(flipped_encoded_state if self.my_series_id == 1 else encoded_state, True)
        self.visualizer.visualize_final_state()
        self.logger.info('Game over: winner=%s end_state=%s, we %s', winner, end_state, "WON" if winner==self.my_series_id else "LOST")

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """Called at the start of each set of games against an opponent
        Args:
            unique_id (int): your unique id within the tournament
            series_id (int): whether you are player 1 or player 2
            player_map (list): (inique_id, series_id) touples for both players
            num_games (int): number of games that will be played
            game_params (list): game-specific parameters.
        Note:
            > For the qualifiers, your player_id should always be "-200",
            but this can change later
            > For Hex, game params will be a 1-length list containing
            the size of the game board ([board_size])
        """
        self.my_series_id = series_id
        self.logger.info(
            'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
            ', game_params=%s',
            unique_id, series_id, player_map, num_games, game_params,
        )

    def state_to_state_encoding(self, state):
        encoded_state = []
        for cell in state[1:]:
            if cell == 1:
                encoded_state.extend((0, 1))
            elif cell == 2:
                encoded_state.extend((1, 0))
            else:
                encoded_state.extend((0, 0))
        # encoded_state.append(state[0] == self.start_player_id)
        # print(state)
        # print(tuple(encoded_state))
        return tuple(encoded_state)

    def encoded_board_flipped(self, encoded_board):
        board_size = 7 # (self.actor.sim_world.get_encoding_shape()[0] - 1) // 14
        flipped_board = [0 for _ in range(board_size * board_size * 2)]
        for row in range(board_size):
            for column in range(board_size):
                index = (row * board_size + column) * 2
                flipped_index = (column * board_size + row) * 2
                cell = (encoded_board[index + 0], encoded_board[index + 1])
                if cell == (1, 0):
                    flipped_board[flipped_index + 0] = 0
                    flipped_board[flipped_index + 1] = 1
                elif cell == (0, 1):
                    flipped_board[flipped_index + 0] = 1
                    flipped_board[flipped_index + 1] = 0
        return tuple(flipped_board)
    
# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()