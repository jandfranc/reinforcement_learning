import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, epsilon, num_machines):
        self.epsilon = epsilon
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = []

        for machine in range(num_machines):
            val = abs(np.random.uniform(0, 10))
            self.actual_rewards.append(val)
            self.machine_list.append(SlotMachine(val))
            self.rewards.append(0)
        self.iteration_list = np.copy(self.rewards)

    def action(self):
        rand_val = np.random.uniform(0.0, 1.0)
        # Explore
        if rand_val < self.epsilon:
            pulled_num = np.random.randint(self.num_machines)
        # Exploit
        else:
            pulled_num = self.rewards.index(max(self.rewards))
        # Get reward
        curr_reward = self.machine_list[pulled_num].pull()
        # Update Mean
        self.iteration_list[pulled_num] += 1
        one_over_iter = 1 / (self.iteration_list[pulled_num])
        self.rewards[pulled_num] = (1 - one_over_iter) * self.rewards[pulled_num] + one_over_iter * curr_reward

        return curr_reward


class SlotMachine:
    def __init__(self, reward_val):
        self.reward_val = reward_val

    def pull(self):
        return np.random.normal(self.reward_val)


if __name__ == "__main__":
    iterations = 10000
    learner1 = BanditAgent(0.1, 5)
    learner2 = BanditAgent(0.01, 5)
    learner3 = BanditAgent(0, 5)
    reward_list1 = []
    reward_list2 = []
    reward_list3 = []
    avg_reward_list1 = []
    avg_reward_list2 = []
    avg_reward_list3 = []
    for iteration in range(0, iterations):
        reward_list1.append(learner1.action())
        reward_list2.append(learner2.action())
        reward_list3.append(learner3.action())
        avg_reward_list1.append(np.mean(reward_list1))
        avg_reward_list2.append(np.mean(reward_list2))
        avg_reward_list3.append(np.mean(reward_list3))
    print(learner1.actual_rewards)
    print(learner1.rewards)
    print(learner2.actual_rewards)
    print(learner2.rewards)
    print(learner3.actual_rewards)
    print(learner3.rewards)
    plt.plot(avg_reward_list1, color='b')
    plt.plot(avg_reward_list2, color='g')
    plt.plot(avg_reward_list3, color='r')
    plt.show()



