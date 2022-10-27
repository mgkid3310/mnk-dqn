import gym
import numpy as np

class mnk_env(gym.Env):
    reward = (-1, 0, 1)

    def __init__(self, m = 3, n = 3, k = 3):
        self.shape = m, n
        self.size = m * n
        self.k = k
        self.turn = 1
        self.done = False

        self.reset()

    def reset(self):
        self.board = np.zeros(self.size, dtype = np.int8)
        self.array = self.board.reshape(self.shape)
        self.observation = np.zeros(self.size, dtype = np.int8)

    def get_obs(self):
        return self.board, self.turn

    def check_result(self, action):
        turn = self.board[action]
        x, y = action // self.shape[0], action % self.shape[0]
        self.array = self.board.reshape(self.shape)

        done, winner = False, 0
        # print((x, y))
        # print(self.array)

        lists = [np.array([], dtype = np.int8)] * 4
        for i in range(1 - self.k, self.k):
            lists[0] = self.append_if_valid(lists[0], x + i, y)
            lists[1] = self.append_if_valid(lists[1], x, y + i)
            lists[2] = self.append_if_valid(lists[2], x + i, y + i)
            lists[3] = self.append_if_valid(lists[3], x + i, y - i)

        for list in lists:
            # print(list)
            i, n = 0, 0
            for i in range(len(list)):
                if list[i] == turn:
                    n += 1
                    if n >= self.k:
                        done, winner = True, turn
                        break
                else:
                    n = 0

        # print((done, winner))

        return done, winner

    def append_if_valid(self, list, x, y):
        if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
            list = np.append(list, self.array[x, y])

        return list

    def step(self, action):
        self.board[action] = self.turn
        self.done, winner = self.check_result(action)
        reward = self.turn * winner

        self.turn = -self.turn

        return self.get_obs(), reward, self.done, None

    def empty_spaces(self):
        return np.where(self.board == 0)

if __name__ == '__main__':
    env = mnk_env(4, 4, 3)
    env.step(1)
    env.step(3)
    env.step(4)
    env.step(6)
    env.step(7)
    env.step(9)
