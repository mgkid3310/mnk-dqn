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
        self.array = self.board.reshape(self.shape)

        self.turn = 1
        self.done = False

    def get_obs(self, visualize = False):
        if visualize:
            return self.array, self.turn
        else:
            return self.board, self.turn

    def get_state(self):
        return torch.tensor(np.array([[self.array * self.turn]])).float()

    def check_result(self, action):
        if np.where(self.board == 0)[0].size == 0:
            return True, 0

        turn = self.board[action]
        x, y = action // self.shape[1], action % self.shape[1]
        self.array = self.board.reshape(self.shape)

        done, winner = False, 0

        lists = [np.array([], dtype = np.int8)] * 4
        for i in range(1 - self.k, self.k):
            lists[0] = self.append_if_valid(lists[0], x + i, y)
            lists[1] = self.append_if_valid(lists[1], x, y + i)
            lists[2] = self.append_if_valid(lists[2], x + i, y + i)
            lists[3] = self.append_if_valid(lists[3], x + i, y - i)

        for list in lists:
            n = 0
            for i in range(len(list)):
                if list[i] == turn:
                    n += 1
                    if n >= self.k:
                        done, winner = True, turn
                        break
                else:
                    n = 0

            if done:
                break

        return done, winner

    def append_if_valid(self, list, x, y):
        if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
            list = np.append(list, self.array[x, y])

        return list

    def step(self, action):
        self.board[action] = self.turn
        self.done, winner = self.check_result(action)
        reward = self.turn * winner

        reward_dict = {1: 1, 0: -1, -1: -1}
        reward = reward_dict[reward]

        self.turn = -self.turn

        return self.get_state(), reward, self.done

    def vaild_actions(self):
        return np.where(self.board == 0)

if __name__ == '__main__':
    env = mnk_env(4, 5, 3)
    env.step(1)
    print(env.get_obs(True))
    env.step(3)
    print(env.get_obs(True))
    env.step(4)
    print(env.get_obs(True))
    env.step(6)
    print(env.get_obs(True))
    env.step(7)
    print(env.get_obs(True))
    env.step(9)
    print(env.get_obs(True))
    env.step(19)
    print(env.get_obs(True))
    env.step(16)
    print(env.get_obs(True))
    env.step(13)
    print(env.get_obs(True))
