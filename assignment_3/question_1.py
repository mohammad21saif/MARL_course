import matplotlib.pyplot as plt
import numpy as np


class MAPFEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.walls = [(5,0), (5,1), (5,2), (4,2),
                      (0,5), (1,5), (2,5), (2,4),
                      (4,9), (4,8), (4,7), (5,7),
                      (7,4), (7,5), (8,5), (9,5)]   

        self.agent_colors = ['blue', 'yellow', 'violet', 'green']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]


    def render(self, agents_pos):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set up gridlines and limits
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)

        # Set aspect of the plot to be equal
        ax.set_aspect('equal')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Remove the axes
        ax.set_xticklabels(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticklabels(np.arange(0, self.grid_size + 1, 1))
        ax.tick_params(left=False, bottom=False)

        for (x,y) in self.walls:
            ax.add_patch(plt.Rectangle((x,y), 1, 1, color='grey'))

        for i, (x,y) in enumerate(agents_pos):
            color = self.agent_colors[i]
            ax.add_patch(plt.Rectangle((x,y), 1, 1, color=color))
            ax.text(x+0.5, y+0.5, str(i), color='black', ha='center', va='center')
        
        for i, (x,y) in enumerate(self.goal_pos):
            color = self.agent_colors[i]
            plt.plot(x+0.5, y+0.5, marker='+', color=color, mew=2, ms=10)

        plt.savefig('mapf_env.png')

mapf_env = MAPFEnv(10)
mapf_env.render(agents_pos=[(1,1), (8,1), (8,8), (1,8)])

