
from matplotlib import pyplot as plt
from simworlds.simworld import SimWorld


class Visualizer:

    def __init__(self, sim_world: SimWorld, frame_delay=0.25):
        self.sim_world = sim_world
        self.visualize = True
        self.frame_delay = frame_delay
    
    def set_visualize(self, value: bool):
        self.visualize = value

    def init_visualize_episode(self, title="Game"):
        if self.visualize:
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.set_window_title(title)

    def visualize_state(self, frame_delay=None):
        if self.visualize:
            self.ax.clear()
            self.sim_world.visualize_state(self.ax)
            plt.pause(frame_delay if frame_delay else self.frame_delay)
            plt.show(block=False)
    
    def visualize_final_state(self):
        self.visualize_state(frame_delay=2*self.frame_delay)
        plt.close()