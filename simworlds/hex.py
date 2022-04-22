import math
from queue import Queue
from xmlrpc.client import boolean

from matplotlib import pyplot as plt
from simworlds.simworld import SimWorld

class HexBoard:

    neighbor_indexes = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    def __init__(self, size):
        self.size = size
        self.init_board()

    def init_board(self):
        self.board = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append((0, 0))
            self.board.append(row)

    def get_neighbors(self, row, column):
        neighbors = []
        for index in self.neighbor_indexes:
            nrow = row + index[0]
            ncolumn = column + index[1]
            if nrow >= 0 and nrow < self.size and ncolumn >= 0 and ncolumn < self.size:
                neighbors.append((nrow, ncolumn))
        return neighbors

    def get_cell(self, row, column):
        return self.board[row][column]

    def set_cell(self, row, column, value):
        self.board[row][column] = value

    def get_board_state(self):
        return self.board

    def get_encoded_board_state(self):
        encoded_list = []
        for row in self.board:
            for cell in row:
                encoded_list.extend(cell)
        return tuple(encoded_list)


class HexGame(SimWorld):

    def __init__(self, size):
        self.size = size
        self.board = None
        self.start_player = True
        self.player_turn = self.start_player # True = "BLACK", False = "RED"
        self.produce_init_state()

    def set_start_player(self, start_player):
        self.start_player = start_player

    def produce_init_state(self):
        self.player_turn = self.start_player
        self.board = HexBoard(self.size)

    def get_action_space(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def get_legal_actions(self):
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                cell = self.board.get_cell(i, j)
                if cell[0] == 0 and cell[1] == 0:
                    actions.append((i, j))
        return actions

    def perform_action(self, action):
        self.board.set_cell(action[0], action[1], (1, 0) if self.player_turn else (0, 1))
        self.player_turn = not self.player_turn

    def bfs(self, start_nodes, end_nodes, node_type):
        visited = set([])
        queue = Queue()
        for node in start_nodes:
            if self.board.get_cell(node[0], node[1]) == node_type:
                queue.put(node)
                visited.add(node)
        while not queue.empty():
            node = queue.get()
            if node in end_nodes:
                return True
            neighbors = self.board.get_neighbors(node[0], node[1])
            for next_node in neighbors:
                if self.board.get_cell(next_node[0], next_node[1]) == node_type and next_node not in visited:
                    queue.put(next_node)
                    visited.add(next_node)
        return False

    def bfs_tree_neighbors(self, start_node, node_type):
        visited = set([])
        queue = Queue()
        result = set([])
        if self.board.get_cell(start_node[0], start_node[1]) == node_type:
            queue.put(start_node)
            visited.add(start_node)
        while not queue.empty():
            node = queue.get()
            neighbors = self.board.get_neighbors(node[0], node[1])
            for next_node in neighbors:
                if self.board.get_cell(next_node[0], next_node[1]) == node_type and next_node not in visited:
                    queue.put(next_node)
                    visited.add(next_node)
                elif self.board.get_cell(next_node[0], next_node[1]) == (0, 0) and not next_node in result:
                    result.add(next_node)
        return list(result)


    def get_neighbors(self, row, column):
        return self.board.get_neighbors(row, column)

    def is_team_winning(self, row_team):
        start_nodes = [(i, 0)             for i in range(self.size)] if row_team else [(0, j)             for j in range(self.size)]
        end_nodes   = [(i, self.size - 1) for i in range(self.size)] if row_team else [(self.size - 1, j) for j in range(self.size)]
        return self.bfs(start_nodes, end_nodes, (1, 0) if row_team else (0, 1))

    def is_final_state(self):
        return self.is_team_winning(True) or self.is_team_winning(False) or len(self.get_legal_actions()) == 0

    def get_reward(self):
        return 1 if self.is_team_winning(True) else -1 if self.is_team_winning(False) else 0 

    def get_current_encoded_state(self):
        state_list = list(self.board.get_encoded_board_state())
        state_list.append(int(self.player_turn))
        # print(tuple(state_list))
        return tuple(state_list)
    
    def get_current_player(self):
        return self.player_turn

    def get_encoding_shape(self):
        return (self.size * self.size * 2 + 1, )

    def set_current_state(self, encoded_state, player):
        for row in range(self.size):
            for column in range(self.size):
                index = (row * self.size + column) * 2
                cell = (encoded_state[index + 0], encoded_state[index + 1])
                self.board.set_cell(row, column, cell)
        self.player_turn = player
    
    def visualize_state(self, ax):
        center_point = (self.size / 2, self.size / 2)
        for i in range(self.size):
            for j in range(self.size):
                cell = self.board.get_cell(i, j)
                neighbors = self.board.get_neighbors(i, j)
                rot_angle = -math.pi / 4
                point = self.rotate(center_point, (j, self.size - i), rot_angle)
                for n_pos in neighbors:
                    # Rotate points
                    n_coord = self.rotate(center_point, (n_pos[1], self.size - n_pos[0]), rot_angle)
                    # Draw Line
                    ax.plot([point[0], n_coord[0]], [point[1], n_coord[1]], color = 'blue', linewidth = 3 - (self.size - 3) * 0.2, linestyle='solid')  
                # Draw Node based on cell type
                color = "r" if cell == (0, 1) else "black" if cell == (1, 0) else "blue"
                circle = plt.Circle(point, 0.2 - (self.size - 3) * 0.01, color=color, fill=not cell==(0, 0))
                ax.add_patch(circle)
        
    
    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy