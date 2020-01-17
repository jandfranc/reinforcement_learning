import numpy as np
import matplotlib.pyplot as plt
import random


class GridWorld:

    def __init__(self, reward_dictionary):
        self.world = np.full([6, 6], 'W')
        self.world[1:5, 1:5] = 'O'
        self.world[4, 1] = 'S'
        self.world[2:4, 2] = 'W'
        self.world[1, 4] = 'G'
        self.world[2, 4] = 'F'
        self.reward_world = np.full([6, 6], reward_dictionary['hit_wall'])
        self.reward_world[1:5, 1:5] = reward_dictionary['none']
        self.reward_world[2:4, 2] = reward_dictionary['hit_wall']
        self.reward_world[1, 4] = reward_dictionary['goal']
        self.reward_world[2, 4] = reward_dictionary['fail']
        self.original_pos = [4, 1]
        self.robot_position = self.original_pos[:]
        print(self.world)

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


class MonteCarloAgent:

    def __init__(self, grid_world, gamma, epsilon):
        self.robot_pos = [0, 0]
        self.original_pos = [0, 0]
        self.potential_pos = self.robot_pos[:]
        self.base_world = np.full([1, 1], float('-inf'))
        self.grid_world = grid_world
        self.movement_list = ['U', 'D', 'L', 'R']
        self.move_list = []
        self.action_expectations = []
        self.gamma = gamma
        self.epsilon = epsilon

    def perform_iteration(self):
        success_str = 'success'
        self.move_list = []
        while success_str != 'reset':
            self.potential_pos = self.robot_pos[:]
            if np.random.uniform(0.0, 1.0) < self.epsilon:
                movement = random.choice(self.movement_list)
            else:
                checker = [index for index in range(0, len(self.action_expectations)) if
                           self.action_expectations[index][0] == self.robot_pos]
                if len(checker) > 0:
                    matched_list = []
                    reward_list = []
                    for item in checker:
                        matched_list.append(self.action_expectations[item][1])
                        reward_list.append(self.action_expectations[item][2])
                    movement = matched_list[reward_list.index(max(reward_list))]
                else:
                    movement = random.choice(self.movement_list)

            reward, success_str = self.grid_world.check_move(movement)

            if movement == 'U':
                self.potential_pos[0] -= 1
            elif movement == 'D':
                self.potential_pos[0] += 1
            elif movement == 'L':
                self.potential_pos[1] -= 1
            elif movement == 'R':
                self.potential_pos[1] += 1

            if success_str == 'success':
                self.robot_pos = self.potential_pos[:]
            self.map_update()
            self.move_list.append((reward, self.robot_pos[:], movement))
            if success_str == 'reset':
                self.robot_pos = self.original_pos[:]

        self.value_update()

    def map_update(self):
        base_y, base_x = self.base_world.shape

        if self.potential_pos[0] == -1:
            self.robot_pos[0] += 1
            self.original_pos[0] += 1
            for item in range(len(self.move_list)):
                self.move_list[item][1][0] += 1
            for item in range(len(self.action_expectations)):
                self.action_expectations[item][0][0] += 1

            base_y += 1
            new_world = np.full([base_y, base_x], float('-inf'))
            new_world[1:base_y, 0:base_x] = self.base_world
            self.base_world = new_world
        elif self.potential_pos[0] == base_y:
            base_y += 1
            new_world = np.full([base_y, base_x], float('-inf'))
            new_world[0:base_y - 1, 0:base_x] = self.base_world
            self.base_world = new_world
        elif self.potential_pos[1] == -1:
            self.robot_pos[1] += 1
            self.original_pos[1] += 1
            base_x += 1
            for item in range(len(self.move_list)):
                self.move_list[item][1][1] += 1
            for item in range(len(self.action_expectations)):
                self.action_expectations[item][0][1] += 1

            new_world = np.full([base_y, base_x], float('-inf'))
            new_world[0:base_y, 1:base_x] = self.base_world
            self.base_world = new_world
        elif self.potential_pos[1] == base_x:
            base_x += 1
            new_world = np.full([base_y, base_x], float('-inf'))
            new_world[0:base_y, 0:base_x - 1] = self.base_world
            self.base_world = new_world

    def value_update(self):
        first_loop = True
        expected_return = 0
        current_updates = []
        reversed_list = reversed(self.move_list)
        for reward, robot_pos, movement in list(reversed_list):
            if first_loop:
                first_loop = False
            else:
                checker = [index for index in range(0, len(self.action_expectations)) if
                           self.action_expectations[index][0] == robot_pos and self.action_expectations[index][
                               1] == movement]
                if len(checker) == 0:
                    self.action_expectations.append((robot_pos, movement, expected_return))
                current_updates.append((robot_pos, expected_return))
            expected_return = reward + self.gamma * expected_return
        for position, expectation in current_updates:
            old_val = self.base_world[position[0], position[1]]
            self.base_world[position[0], position[1]] = max([old_val, expectation])


if __name__ == "__main__":
    reward_dict = {'none': 0,
                   'hit_wall': -0.5,
                   'goal': 1,
                   'fail': -1}
    world = GridWorld(reward_dict)
    MC_agent = MonteCarloAgent(world, 0.9, 0.1)
    for i in range(100):
        MC_agent.perform_iteration()
    print(np.max(MC_agent.base_world))
    plt.imshow(MC_agent.base_world)
    plt.show()
