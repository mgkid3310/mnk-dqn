#%%
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D

class DQN_player():

    def __init__(self):
        self.name = "DQN_player"
        self.epsilon = 1
        self.learning_rate = 0.1
        self.gamma=0.9

        self.main_network = self.make_network()
        self.target_network = self.make_network()
        self.copy_network() # 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사

        self.count = np.zeros(25)
        self.win = np.zeros(25)
        self.begin = 0
        self.e_trend = []

    def make_network(self):
        """
        신경망 생성
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(5,5,4)))
        self.model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(25))
        # print(self.model.summary())

        self.model.compile(optimizer = SGD(learning_rate=0.01), loss = 'mean_squared_error', metrics=['mse'])

        return self.model

    def copy_network(self):
        """
        신경망 복사
        """
        self.target_network.set_weights(self.main_network.get_weights())

    def save_network(self, name):
        filename = name + '_main_network.h5'
        self.main_network.save(filename)
        print("end save model")

    def state_convert(self, board_a):
        """
        1차원 배열의 보드상태를 2차원으로 변환
        """
        d_state = np.full((5,5,4),0.1)
        for i in range(25):
            if board_a[i] == 1:
                d_state[i//5,i%5,0] = 1
            elif board_a[i] == -1:
                d_state[i//5,i%5,1] = 1
            else:
                pass
        return d_state

    def policy(self, env):
        available_state = env.get_action() # 행동 가능한 상태를 저장

        state_2d = self.state_convert(env.board_a)
        x = np.array([state_2d],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x, verbose=0)[0,:]

        available_state_qvalues = qvalues[available_state] # 행동 가능한 상태의 Q-value를 저장

        greedy_action = np.argmax(available_state_qvalues) # max Q-value를 탐색한 후 저장

        # max Q-value와 같은 값이 여러개 있는지 확인한 후 double_check에 상태를 저장
        double_check = (np.where(qvalues == np.max(available_state[greedy_action]),1,0))

        #  여러개 있다면 중복된 상태중에서 다시 무작위로 선택    
        if np.sum(double_check) > 1:
            double_check = double_check / np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)

        # ε-greedy
        pr = np.zeros(len(available_state))

        for i in range(len(available_state)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],4)))
            else:
                pr[i] = self.epsilon / len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],4)))

        action = np.random.choice(range(0,len(available_state)), p=pr)        

        if len(available_state) == 25:
            self.count[action] +=1
            self.begin = action

        return available_state[action]

    def learn_dqn(self,board_bakup, action_backup, env, reward):

        # 입력을 2차원으로 변환한 후, 메인 신경망으로 q-value를 계산
        new_state = self.state_convert(board_bakup)
        x = np.array([new_state],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x, verbose=0)[0,:]
        delta = 0

        if env.done == True:
            if reward == 1:
                self.win[self.begin] += 1

            # 게임이 좀료됐을때 신경망의 학습을 위한 정답 데이터를 생성
            qvalues[action_backup] = reward
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            # 생성된 정답 데이터로 메인 신경망을 학습
            self.main_network.fit(x, y, epochs=10, verbose=0)

        else:
            # 게임이 진행중일때  신경망의 학습을 위한 정답 데이터를 생성
            # 현재 상태에서 최고 Q 값을 계산
            new_state = self.state_convert(env.board_a)
            next_x = np.array([new_state],dtype=np.float32).astype(np.float32)
            next_qvalues = self.target_network.predict(next_x, verbose=0)[0,:]
            available_state = env.get_action()
            maxQ = np.max(next_qvalues[available_state])

            delta = self.learning_rate*(reward + self.gamma * maxQ - qvalues[action_backup])
            qvalues[action_backup] += delta
            # 생성된 정답 데이터로 메인 신경망을 학습
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            self.main_network.fit(x, y, epochs = 10, verbose=0)
