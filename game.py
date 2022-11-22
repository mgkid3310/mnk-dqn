import numpy as np
import torch

class mnk_env():
    reward = (-1, 0, 1)

    def __init__(self, m = 3, n = 3, k = 3):
        self.shape = m, n
        self.format = m, n, 1
        self.size = m * n
        self.k = k

        self.reset()

    def reset(self):
        self.board = np.zeros(self.size, dtype = np.int8)

        self.turn = 1
        self.win_in_one = 0
        self.blocks = []
        self.winner = 0
        self.done = False

    def get_obs(self, visualize = False):
        if visualize:
            return self.board.reshape(self.shape), self.turn
        else:
            return self.board, self.turn

    def get_state(self):
        array = self.board.reshape(self.shape)
        return torch.tensor(np.array([[array * self.turn]])).float()

    def check_result(self, board, action, turn):
        if np.where(board == 0)[0].size == 0:
            return True, 0

        x, y = action // self.shape[1], action % self.shape[1]

        lists = [np.array([], dtype = np.int8)] * 4
        for i in range(1 - self.k, self.k):
            lists[0] = self.append_if_valid(lists[0], x + i, y)
            lists[1] = self.append_if_valid(lists[1], x, y + i)
            lists[2] = self.append_if_valid(lists[2], x + i, y + i)
            lists[3] = self.append_if_valid(lists[3], x + i, y - i)

        done, winner = self.check_win(lists, turn)

        return done, winner

    def check_win(self, lists, turn):
        done, winner = False, 0

        for list in lists:
            done, winner = self.check_streak(list, turn)
            if done:
                break

        return done, winner

    def check_win_in_one(self, turn, done):
        if done:
            return 0, []

        actions = self.vaild_actions()
        win_in_one, blocks = 0, []
        for action in actions:
            board_copy = self.board.copy()
            board_copy[action] = -turn

            done, winner = self.check_result(board_copy, action, -turn)
            if done:
                win_in_one = winner
                blocks.append(action)

        return win_in_one, blocks

    def check_streak(self, list, turn):
        if self.k > len(list):
            return False, 0

        n = 0
        for i in range(len(list)):
            if list[i] == turn:
                n += 1
                if n >= self.k:
                    return True, turn
            else:
                n = 0

        return False, 0

    def append_if_valid(self, list, x, y):
        if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
            list = np.append(list, self.board.reshape(self.shape)[x, y])

        return list

    def step(self, action):
        self.board[action] = self.turn
        self.done, self.winner = self.check_result(self.board, action, self.turn)
        self.win_in_one, self.blocks = self.check_win_in_one(self.turn, self.done)

        if self.done:
            reward = self.turn * self.winner
        else:
            reward = self.turn * self.win_in_one * len(self.blocks)
            self.turn = -self.turn

        return self.get_state(), reward, self.done

    def vaild_actions(self):
        return np.where(self.board == 0)[0]

if __name__ == '__main__':
    env = mnk_env(5, 5, 3)
    done = False
    while not done:
        state, reward, done = env.step(int(input()))
        print(env.get_obs(True))
