import bayesian_sampling as bs
import epsilon_greedy as eg
import optimistic_initial_values as oiv
import UCB1 as ucb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    actual_rewards = [1, 2, 3, 4, 5]
    iterations = 10000

    learner1 = bs.BanditAgent(5, actual_rewards)
    learner2 = eg.BanditAgent(0.01, 5, actual_rewards)
    learner3 = oiv.BanditAgent(10, 5, actual_rewards)
    learner4 = ucb.BanditAgent(5, actual_rewards)

    avg_rewards1 = bs.run_experiment(learner1, iterations)
    avg_rewards2 = bs.run_experiment(learner2, iterations)
    avg_rewards3 = bs.run_experiment(learner3, iterations)
    avg_rewards4 = bs.run_experiment(learner4, iterations)

    print(learner1.actual_rewards)
    print(learner1.rewards)
    print(learner2.actual_rewards)
    print(learner2.rewards)
    print(learner3.actual_rewards)
    print(learner3.rewards)
    print(learner4.actual_rewards)
    print(learner4.rewards)

    plt.plot(avg_rewards1, color='b')
    plt.plot(avg_rewards2, color='g')
    plt.plot(avg_rewards3, color='r')
    plt.plot(avg_rewards4, color='k')
    # plt.xscale('log')
    plt.show()