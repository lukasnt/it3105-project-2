import json
from learners.anet import ANET_Parameters, ActorNeuralNetwork
from learners.dtrees import DecisionTrees, DecisionTreesParams
from learners.learner import Learner
from rl import RLSystem
from simworlds.hex import HexGame
from simworlds.nim import Nim
from simworlds.simworld import SimWorld
from topp import TOPP

file = open("params.json")
params = json.load(file)
file.close()

sim_world = SimWorld()
sw_name = params["simworld"]
sw_params = params["simworld_params"]
if sw_name == "nim":
    nim_params = sw_params["nim"]
    sim_world = Nim(nim_params["start"], nim_params["max_move"], True)
elif sw_name == "hex":
    hex_params = sw_params["hex"]
    sim_world = HexGame(hex_params["board_size"])

learner = Learner()
learner_name = params["learner"]
learner_params = params["learner_params"]
if learner_name == "anet":
    anet_params = learner_params["anet"]
    anet_params = ANET_Parameters(sim_world.get_encoding_shape(), sim_world.get_action_space(), anet_params["dimensions"], anet_params["learning_rate"], anet_params["activation"], anet_params["optimizer"])
    learner = ActorNeuralNetwork(anet_params)
elif learner_name == "dtrees":
    dtrees_params = DecisionTreesParams(len(sim_world.get_action_space()))
    learner = DecisionTrees(dtrees_params)

topp = TOPP(sim_world, params["TOPP_players"], params["TOPP_games"], params["episodes"], params["search_games"], learner, train_visualize=params["train_visualize"], tournament_visualize=params["TOPP_visualize"], frame_delay=params["frame_delay"], train_epsilon=params["epsilon"])

if params["train_enabled"]:
    topp.train_players()
    topp.save_params(params)

# Ids: 1648719880, 1648720704, 1648721304, 1648731768, 1648732720, 1649060090
if params["TOPP_restore_players"]:
    topp.restore_trained_players(params["TOPP_restore_players"])

if params["TOPP_enabled"]:
    topp.play_tournament()


# rl = RLSystem(sim_world)
# rl.run_episodes(20, 1000)