from time import time
import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorld:

    def __init__(self, reward_dictionary):
        self.world = np.full([5, 6], 'W')
        self.world[1:4, 1:5] = 'O'
        self.world[3, 1] = 'S'
        self.world[2, 2] = 'W'
        self.world[1, 4] = 'G'
        self.world[2, 4] = 'F'
        self.reward_world = np.full([5, 6], reward_dictionary['hit_wall'])
        self.reward_world[1:4, 1:5] = reward_dictionary['none']
        self.reward_world[2:4, 2] = reward_dictionary['hit_wall']
        self.reward_world[1, 4] = reward_dictionary['goal']
        self.reward_world[2, 4] = reward_dictionary['fail']
        self.original_pos = [3, 1]
        self.robot_position = self.original_pos[:]

    def check_move(self, movement):
        previous_pos = self.robot_position[:]
        if movement == 'U':
            self.robot_position[0] -= 1
        elif movement == 'D':
            self.robot_position[0] += 1
        elif movement == 'L':
            self.robot_position[1] -= 1
        elif movement == 'R':
            self.robot_position[1] += 1

        current_tile = self.world[self.robot_position[0], self.robot_position[1]]
        reward = self.reward_world[self.robot_position[0], self.robot_position[1]]

        if current_tile == 'W':
            self.robot_position = previous_pos[:]
            success_str = 'fail'
        elif current_tile == 'G':
            self.robot_position = self.original_pos[:]
            success_str = 'reset'
        elif current_tile == 'F':
            self.robot_position = self.original_pos[:]
            success_str = 'reset'
        else:
            success_str = 'success'

        return reward, success_str


class SARSAAgent:

    def __init__(self, grid_world, gamma, above_epsilon):
        self.robot_pos = [0, 0]
        self.original_pos = [0, 0]
        self.grid_world = grid_world
        self.movement_list = ['U', 'D', 'L', 'R']
        self.move_list = []
        self.action_expectations = {}
        self.state_expectations = {}
        self.gamma = gamma
        self.above_epsilon = above_epsilon
        self.possible_positions = []
        self.theta = np.random.randn(25) / np.sqrt(25)
        self.deltas = []
        self.biggest_change = 0

    def sa2x(self, s, a):
        return np.array([
            s[0] - 1 if a == 'U' else 0,
            s[1] - 1.5 if a == 'U' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'U' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'U' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0,
            1 if a == 'U' else 0,
            s[0] - 1 if a == 'D' else 0,
            s[1] - 1.5 if a == 'D' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'D' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'D' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0,
            1 if a == 'D' else 0,
            s[0] - 1 if a == 'L' else 0,
            s[1] - 1.5 if a == 'L' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'L' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'L' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0,
            1 if a == 'L' else 0,
            s[0] - 1 if a == 'R' else 0,
            s[1] - 1.5 if a == 'R' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'R' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'R' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0,
            1 if a == 'R' else 0,
            1
        ])

    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)

    def get_state_dict(self, state):
        state_dict = {}
        for action in self.movement_list:
            state_dict[action] = [self.predict(state, action), 0]
        return state_dict

    def perform_iteration(self):
        self.biggest_change = 0
        success_str = 'success'
        first_loop = True
        loop_check = 0
        while success_str != 'reset':
            loop_check += 1

            movement = self.choose_move_e_greedy()

            reward, success_str = self.grid_world.check_move(movement)
            reward = reward - 0.1
            if not first_loop:
                update_move_list = next_move_list[:]
                next_move_list = [reward, self.robot_pos[:], movement]
                self.new_value_update(next_move_list, update_move_list)
            else:
                next_move_list = [reward, self.robot_pos[:], movement]
                first_loop = False

            if success_str == 'success':
                if movement == 'U':
                    self.robot_pos[0] -= 1
                elif movement == 'D':
                    self.robot_pos[0] += 1
                elif movement == 'L':
                    self.robot_pos[1] -= 1
                elif movement == 'R':
                    self.robot_pos[1] += 1

            if success_str == 'reset':
                self.final_value_update(reward, next_move_list)
                self.robot_pos = self.original_pos[:]
        self.deltas.append(self.biggest_change)
    def choose_move_e_greedy(self):
        previous_reward = float('-inf')
        action = None
        reward_list = []
        random.shuffle(self.movement_list)
        if random.uniform(0, 1) > self.above_epsilon:
            action_dict = self.get_state_dict(self.robot_pos)
            for key in action_dict:
                value = action_dict[key][0]
                if value > previous_reward:
                    previous_reward = value
                    action = key

        else:
            action = random.choice(self.movement_list)
        return action

    def new_value_update(self, next_move_list, update_move_list):

        if str(update_move_list[1]) not in self.action_expectations:
            self.action_expectations[str(update_move_list[1])] = self.get_state_dict(update_move_list[1])
            self.possible_positions.append(update_move_list[1])

        if str(next_move_list[1]) not in self.action_expectations:
            self.action_expectations[str(next_move_list[1])] = self.get_state_dict(next_move_list[1])
            self.possible_positions.append(next_move_list[1])

        self.action_expectations[str(update_move_list[1])][update_move_list[2]][1] += 1

        step_size = 1 / self.action_expectations[str(update_move_list[1])][update_move_list[2]][1]
        old_theta = self.theta.copy()
        self.theta += step_size * (
                next_move_list[0] + self.gamma * self.predict(next_move_list[1], next_move_list[2]) -
                self.predict(update_move_list[1], update_move_list[2])) * \
                      self.grad(update_move_list[1], update_move_list[2])
        self.biggest_change = max(self.biggest_change, np.abs(self.theta - old_theta).sum())

    def final_value_update(self, reward, update_move_list):
        if str(update_move_list[1]) not in self.action_expectations:
            self.action_expectations[str(update_move_list[1])] = {}
            for move in self.movement_list:
                self.action_expectations[str(update_move_list[1])][move] = [self.predict(update_move_list[1], move), 0]

        self.action_expectations[str(update_move_list[1])][update_move_list[2]][1] += 1

        step_size = 1 / self.action_expectations[str(update_move_list[1])][update_move_list[2]][1]
        old_theta = self.theta.copy()
        self.theta += step_size * (reward - self.predict(update_move_list[1], update_move_list[2])) * \
            self.grad(update_move_list[1], update_move_list[2])
        self.biggest_change = max(self.biggest_change, np.abs(self.theta - old_theta).sum())

    def get_policy(self):
        policy_dict = {}
        for position in self.possible_positions:
            best_action = None
            best_reward = float('-inf')
            for move in self.movement_list:
                reward = self.action_expectations[str(position)][move][0]
                if reward > best_reward:
                    best_reward = reward
                    best_action = move
                policy_dict[str(position)] = best_action
        return policy_dict


if __name__ == "__main__":
    reward_dict = {'none': 0,
                   'hit_wall': -0.5,
                   'goal': 1,
                   'fail': -1}
    world = GridWorld(reward_dict)
    SARSA_agent = SARSAAgent(world, 0.9, 0.1)
    t = time()
    for i in range(500):
        SARSA_agent.perform_iteration()

    plt.plot(SARSA_agent.deltas)
    plt.show()
    print(time() - t)
    print(SARSA_agent.get_policy())
