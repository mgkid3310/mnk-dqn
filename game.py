import gym
import numpy as np

class mnk_env(gym.Env):
    reward = (-1, 0, 1)

    def __init__(self, m = 3, n = 3, k = 3):
        self.size = m * n
        self.k = k
        self.turn = 1
        self.done = False

        self.reset()

    def reset(self):
        self.board = np.zeros(self.size, dtype = np.int8)
        self.observation = np.zeros(self.size, dtype = np.int8)

    def get_obs(self):
        return self.board, self.turn

    def check_result(self):
        done = False
        winner = 0

        return done, winner

    def step(self, action):
        self.board[action] = self.turn
        self.done, winner = self.check_result()
        reward = self.turn * winner

        self.turn = -self.turn

        return self.get_obs(), reward, self.done, None

    def empty_spaces(self):
        return np.where(self.board == 0)

if __name__ == '__main__':
    env = mnk_env(3, 3, 3)
    print(env.step(1))
    print(env.step(3))
