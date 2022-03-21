from mcts import MonteCarloTreeSearch
from simworlds.nim import Nim


sim_world = Nim(11, 4, True)
mcts = MonteCarloTreeSearch(sim_world)

mcts.run_episodes(20, 500)