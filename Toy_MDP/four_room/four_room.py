# import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from celluloid import Camera
import time



class FourRooms:
    """
    A toy grid world example:

    Environment instance:
        Rewards are distributed with goal state at G
    """

    def __init__(self, goal_state, subgoals, 
                nrows=11, ncols=11, 
                B=200.0, noise=None):
        self.nrows = nrows
        self.ncols = ncols 
        self.noise = noise
        self.SUBGOALS = subgoals
        self.START_STATE = [0, 0]
        if self.noise is not None:
            self.mu, self.sigma = self.noise
        self.S1, self.S2, self.S3, self.S4 = self.SUBGOALS
        self.GOAL_STATE = goal_state
        self.B = B
        self.subgoal_reward = +225
        self.goal_reward = 1


        # define walls of the rooms
        self.wall_1 = [[k, 5] for k in range(self.nrows)]
        self.wall_2 = [[k, 5] for k in range(5, self.ncols)]
        self.wall_3 = [[5, k] for k in range(5)]
        self.wall_4 = [[6, k] for k in range(5, self.ncols)]
        self.wall_types = self.wall_1 + self.wall_2 + self.wall_3 + self.wall_4
        self.walls = [k for k in self.wall_types if k not in self.SUBGOALS]         

        # Define action space: four actions ->
        # up, down, left, right
        self.actions = [[-1, 0], [+1, 0], [0, -1], [0, +1]]
        self.action_space = len(self.actions)
        self.rewards, self.Y = self.rewards_distribution()
        self.reset()   # reset to starting state

    def _get_obs(self):
        return self.state

    def sample_action(self):
        action_id = np.random.choice(self.action_space)
        return action_id

    def reset(self):
        self.state = [0, 0]
        return np.array(self._get_obs())

    def rewards_distribution(self):
        x1c, x2c = self.GOAL_STATE
        x1 = np.arange(0, self.nrows, 1)
        x2 = np.arange(0, self.ncols, 1)
        X1, X2 = np.meshgrid(x1, x2)
        Y = self.B - ((X1 - x1c) ** 2 + (X2 - x2c) ** 2)

        # add gaussian noise if the reward is stochastic
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.noise is not None:
                    Y[i, j] = Y[i, j] + np.random.normal(self.mu, self.sigma)
                else:
                    Y[i, j] = Y[i, j]

        rewards = np.copy(Y)
        rewards = np.ones(Y.shape) * 0.0
        # walls of rooms
        for k in self.walls:
            rewards[k[0], k[1]] = -1

        # # options (hallway subgoals)
        # rewards[self.S1[0], self.S1[1]] = self.subgoal_reward
        # rewards[self.S2[0], self.S2[1]] = self.subgoal_reward
        # rewards[self.S3[0], self.S3[1]] = self.subgoal_reward
        # rewards[self.S4[0], self.S4[1]] = self.subgoal_reward
        
        # modify goal state
        # rewards[self.GOAL_STATE[1], self.GOAL_STATE[0]] = self.goal_reward
        # rewards /= np.max(rewards) 
        return rewards, Y

    def step(self, action):
        
        action_val = self.actions[action]

        next_state = [self.state[0] + action_val[0], self.state[1] + action_val[1]]
        
        if next_state not in self.walls:
            self.state[0] = min(max(self.state[0] + action_val[0], 0), 10)
            self.state[1] = min(max(self.state[1] + action_val[1], 0), 10)

        reward = self.rewards[self.state[0], self.state[1]]

        # check if next state is the terminal state
        if self.state == self.GOAL_STATE:
            done = True
        else:
            done = False
        
        if done: 
            reward = 0
        else:
            reward = -1

        next_state = self._get_obs()
        return np.array(next_state), reward, done


class Visualizations(FourRooms):
    def __init__(self, goal_state, subgoals, 
                nrows=11, ncols=11, 
                B=200.0, noise=None):
        super().__init__(goal_state, subgoals, 
                nrows=11, ncols=11, 
                B=200.0, noise=None)
        x1 = np.arange(0, self.nrows, 1)
        x2 = np.arange(0, self.ncols, 1)
        self.X1, self.X2 = np.meshgrid(x1, x2)
        # ----------------------------------------------------------------------------------
        # Use latex font for each plot, comment this section out if latex is not supported
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # -----------------------------------------------------------------------------------
        self.rewards, self.Y = self.rewards_distribution()

    def plot_reward_distribution(self, plot_type=None, save=False):        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca(projection='3d')

        # ---------------- plot 3D reward bar --------------
        if plot_type == 'bar':
            rewards = self.Y.ravel()
            _X = self.X1.ravel()
            _Y = self.X2.ravel()
            top = rewards
            bottom = np.zeros_like(top)
            width = depth = 1
            cmap = cm.coolwarm
            rgba = [cmap((k - np.min(top)) / np.max(top)) for k in top]
            ax.bar3d(_X, _Y, bottom, width, depth, top, color=rgba, shade=True)
            # ax.view_init(25, -145)
            # plt.draw()
        else:
            # Plot the surface.
            surf = ax.plot_surface(self.X1, self.X2, Y, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.tick_params(axis='both', which='major', labelsize=20)

        if save:
            plt.savefig("figures/reward.pdf", dpi=300, bbox_inches='tight')
        plt.show()


    def four_rooms_viz(self, fig_name=None, save=False, fignum=1):
        # Set every cell to reward value
        image = self.rewards
        # Reshape things into a (nrows x ncols) grid.
        image = image.reshape((self.nrows, self.ncols))

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(self.nrows)
        col_labels = range(self.ncols)
        plt.matshow(image)
        plt.xticks(range(self.ncols), col_labels)
        plt.yticks(range(self.nrows), row_labels)
        plt.scatter(self.GOAL_STATE[1], self.GOAL_STATE[0], s=750, marker='*', color='blue')
        plt.scatter(self.START_STATE[1], self.START_STATE[0], s=500, marker='o', color='blue')
        if save is not None and fig_name is not None:
            plt.savefig("figures/{}.pdf".format(fig_name),
                        dpi=300, bbox_inches='tight')
        # plt.show()


    def policy_visualization(self, states, actions, save=False, fig_name=None, fignum=1):
        # Make a Y.shape grid...
        nrows, ncols = self.rewards.shape
        # Set every cell to reward value
        image = self.rewards
        # Reshape things into a (nrows x ncols) grid.
        image = image.reshape((nrows, ncols))

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(nrows)
        col_labels = range(ncols)
        plt.matshow(image)
        plt.xticks(range(ncols), col_labels)
        plt.yticks(range(nrows), row_labels)
        plt.scatter(self.GOAL_STATE[1], self.GOAL_STATE[0], s=750, marker='*', color='blue')
        plt.scatter(self.START_STATE[1], self.START_STATE[0], s=500, marker='o', color='blue')
        # plot policy
        x_pos = np.zeros(len(actions))
        y_pos = np.zeros(len(actions))
        for i in range(len(actions)):
            x_pos[i] = states[i][1]
            y_pos[i] = states[i][0]
            if actions[i] == self.actions[0]:  # UP
                x, y = x_pos[i], y_pos[i]+0.4
                dx, dy = 0, -0.8
            elif actions[i] == self.actions[1]:  # DOWN
                x, y = x_pos[i], y_pos[i]-0.4
                dx, dy = 0, 0.8
            elif actions[i] == self.actions[2]:  # LEFT
                x, y = x_pos[i]+0.4, y_pos[i]
                dx, dy = -0.8, 0
            elif actions[i] == self.actions[3]:  # RIGHT
                x, y = x_pos[i]-0.4, y_pos[i]
                dx, dy = 0.8, 0
            else:
                raise Exception('Invalid action!')

            plt.arrow(x, y, dx, dy, length_includes_head=True,
                      head_width=0.15, head_length=0.3, lw=2.0, color='black')
        if save and fig_name is not None:
            plt.savefig("figures/{}.pdf".format(fig_name),
                        dpi=300, bbox_inches='tight')
        # plt.show()


    def policy_animatation(self, states, save=False, fig_name=None):
        image = np.zeros(self.nrows * self.ncols)
        # Set every other cell to a random number (this would be your data)
        image = self.rewards
        # Reshape things into a 9x9 grid.
        image = image.reshape((self.nrows, self.ncols))
        row_labels = range(self.nrows)
        col_labels = range(self.ncols)
        plt.matshow(image)
        plt.xticks(range(self.ncols), range(self.ncols))
        plt.yticks(range(self.nrows), range(self.nrows))
        plt.scatter(self.GOAL_STATE[1], self.GOAL_STATE[0], s=750, marker='*', color='blue')
        plt.scatter(self.START_STATE[1], self.START_STATE[0], s=500, marker='o', color='blue')

        camera = Camera(plt.figure(figsize=(10, 8)))

        # policy
        x = [states[i][1] for i in range(len(states))]
        y = [states[i][0] for i in range(len(states))]

        for i in range(len(x)):
            xdata, ydata = x[:i + 1], y[:i + 1]
            plt.plot(xdata, ydata, '-', color='white', markersize=10, lw=2.0)
            plt.plot(xdata[-1], ydata[-1], 'o', color='red', markersize=10, lw=2.0)
            # Make a Y.shape grid...
            nrows, ncols = self.Y.shape
            image = np.zeros(nrows * ncols)
            # Set every other cell to a random number (this would be your data)
            image = self.rewards
            # Reshape things into a 9x9 grid.
            image = image.reshape((self.nrows, self.ncols))
            row_labels = range(self.nrows)
            col_labels = range(self.ncols)
            plt.matshow(image, fignum=0)  # matshow: gridworld
            plt.xticks(range(self.ncols), col_labels)
            plt.yticks(range(self.nrows), row_labels)
            plt.scatter(self.GOAL_STATE[1], self.GOAL_STATE[0], s=750, marker='*', color='blue')
            plt.scatter(self.START_STATE[1], self.START_STATE[0], s=500, marker='o', color='blue')

            camera.snap()

        anim = camera.animate(blit=True)
        if save and fig_name is not None:
            anim.save(f'figures/{fig_name}.gif', writer='imagemagick')
        # plt.show()



