import numpy as np
import matplotlib.pyplot as plt


class TicTacToe:

    def __init__(self):
        self.board = np.full((3, 3), 0)
        self.win_con_1 = [2, 2, 2]
        self.win_con_2 = [1, 1, 1]

    def clear_board(self):
        self.board = np.full((3, 3), 0)

    def fill_spot(self, turn, position):
        self.board[position[0], position[1]] = turn

    def check_win(self):
        for idx in range(3):
            row = self.board[idx, :]
            col = np.transpose(self.board[:, idx])
            if np.all(row == self.win_con_1) or np.all(row == self.win_con_2) \
                    or np.all(col == self.win_con_1) or np.all(col == self.win_con_2):
                return True, 'end'
        dia_1 = self.board.diagonal()
        dia_2 = np.fliplr(self.board).diagonal()

        if np.all(dia_1 == self.win_con_1) or np.all(dia_1 == self.win_con_2) \
                or np.all(dia_2 == self.win_con_1) or np.all(dia_2 == self.win_con_2):
            return True, 'end'
        if np.all(self.board != 0):
            return True, 'draw'
        return False, 'none'


class Learner:

    def __init__(self, epsilon, step_size):
        self.win = 1
        self.draw = 0
        self.lose = 0
        self.epsilon = epsilon
        self.step_size = step_size
        self.state_dict = {}
        self.action_dict = {}

    def update_value_function(self, current_state, next_state=[1], reward=0):
        if np.shape(next_state) != (3, 3):
            self.state_dict[str(current_state)] = reward
        else:
            if str(current_state) not in self.state_dict:
                self.state_dict[str(current_state)] = 0
            if str(next_state) not in self.state_dict:
                self.state_dict[str(current_state)] = 0
            self.state_dict[str(current_state)] = self.state_dict[str(current_state)] + \
                                                  self.step_size * (self.state_dict[str(next_state)] - \
                                                                    self.state_dict[str(current_state)])

    def update_value_iterator(self, states, reward):
        states.reverse()
        for iteration, state in enumerate(states):
            if iteration == 0:
                self.update_value_function(state, reward=reward)
                next_state = state
            else:
                self.update_value_function(state, next_state, reward)
                next_state = state

    def create_action_dict(self, current_state, turn):
        row, col = np.shape(current_state)
        self.action_dict[str(current_state)] = []
        for i_row in range(row):
            for i_col in range(col):
                if current_state[i_row, i_col] == 0:
                    next_state = np.copy(current_state)
                    next_state[i_row, i_col] = turn
                    self.action_dict[str(current_state)].append(next_state)
                    if str(next_state) not in self.state_dict:
                        self.state_dict[str(next_state)] = 0

    def choose_move(self, current_state, turn):
        if str(current_state) not in self.action_dict:
            self.create_action_dict(current_state, turn)
        rand_val = np.random.uniform(0.0, 1.0)
        # Explore
        if rand_val < self.epsilon:
            # print(len(self.action_dict[str(current_state)]))
            move = self.action_dict[str(current_state)][np.random.randint(0, len(self.action_dict[str(current_state)]))]
        # Exploit
        else:
            state_values = []
            for states in self.action_dict[str(current_state)]:
                state_values.append(self.state_dict[str(states)])
            # print(state_values)
            move = self.action_dict[str(current_state)][state_values.index(max(state_values))]

        return move


def play_game(tictactoe, learner0, learner1):
    iter_states = []
    game_end = False
    turn = 0
    placers = [1, 2] * 5
    while not game_end:
        if turn == 0:
            tictactoe.board = learner0.choose_move(tictactoe.board, placers.pop())

            turn = 1
        else:
            tictactoe.board = learner1.choose_move(tictactoe.board, placers.pop())
            turn = 0

        iter_states.append(tictactoe.board)
        game_end, end_type = tictactoe.check_win()

    if end_type == 'draw':
        learner0.update_value_iterator(iter_states, reward=0)
        learner1.update_value_iterator(iter_states, reward=0)
    elif turn == 1:
        learner0.update_value_iterator(iter_states, reward=1)
        learner1.update_value_iterator(iter_states, reward=-1)
    else:
        learner0.update_value_iterator(iter_states, reward=-1)
        learner1.update_value_iterator(iter_states, reward=1)


def human_input(board, turn):
    board_copy = np.copy(board)
    choice = int(input('enter num pls'))
    choice1 = int(input('enter num pls'))
    board_copy[choice, choice1] = turn
    return board_copy


def play_game_human(tic_tac_toe, learner_0, start):
    iter_states = []
    game_end = False
    turn = start
    placers = [1, 2] * 5
    game.board = np.full((3, 3), 0)
    while not game_end:
        if turn == 0:
            if str(tic_tac_toe.board) in learner_0.action_dict:
                for states in learner_0.action_dict[str(tic_tac_toe.board)]:
                    if str(states) in learner_0.state_dict:
                        print(learner_0.state_dict[str(states)])
                    else:
                        print(-10)
            tic_tac_toe.board = learner_0.choose_move(tic_tac_toe.board, placers.pop())
            turn = 1
        else:
            tic_tac_toe.board = human_input(tic_tac_toe.board, placers.pop())
            turn = 0
        print(tic_tac_toe.board)
        game_end, end_type = tic_tac_toe.check_win()
    print(end_type)


if __name__ == '__main__':
    game = TicTacToe()
    learner0 = Learner(0.1, 0.1)
    learner1 = Learner(0.1, 0.1)
    for i in range(100000):
        play_game(game, learner0, learner1)
        game.clear_board()
        '''
        plt.imshow(game.board)
        plt.ion()
        plt.show()
        plt.pause(2)
        '''
    for i in range(1000):
        play_game_human(game, learner0, 0)
        play_game_human(game, learner1, 1)
