import numpy as np

class Q_learning_player():

    def __init__(self):
        self.name = "Q_player"
        self.qtable = {} # Q-table을 딕셔너리로 정의
        self.epsilon = 0.5 # e-greedy 계수
        self.learning_rate = 0.1 # 학습률
        self.gamma=0.9

    def select_action(self, env):
        """
        policy에 따라 상태에 맞는 행동을 선택
        """
        action = self.policy(env)
        return action

    def policy(self, env):
        available_action = env.get_action() # 행동 가능한 상태를 저장
        qvalues = np.zeros(len(available_action)) # 행동 가능한 상태의 Q-value를 저장

        # 행동 가능한 상태의 Q-value를 조사
        for i, act in enumerate(available_action):
            key = (tuple(env.board_a),act)
            # 현재 상태를 경험한 적이 없다면(딕셔너리에 없다면) 딕셔너리에 추가(Q-value = 0)
            if self.qtable.get(key) ==  None:
                self.qtable[key] = 0
            # 행동 가능한 상태의 Q-value 저장
            qvalues[i] = self.qtable.get(key)

        # e-greedy
        # 가능한 행동들 중에서 Q-value가 가장 큰 행동을 저장
        greedy_action = np.argmax(qvalues)

        pr = np.zeros(len(available_action))

        # max Q-value와 같은 값이 여러개 있는지 확인한 후 double_check에 상태를 저장
        double_check = (np.where(qvalues == np.max(qvalues),1,0))

        #  여러개 있다면 중복된 상태중에서 다시 무작위로 선택
        if np.sum(double_check) > 1:
            double_check = double_check/np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)
        # e-greedy로 행동들의 선택 확률을 계산
        pr = np.zeros(len(available_action))

        for i in range(len(available_action)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_action)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))
            else:
                pr[i] = self.epsilon / len(available_action)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))
        action = np.random.choice(range(0,len(available_action)), p=pr)

        return available_action[action]

    def learn_qtable(self,board_bakup, action_backup, env, reward):
        key = (board_bakup,action_backup) # 현재 상태와 행동을 키로 저장

        if env.done == True: # 게임이 끝났을 경우 학습
            self.qtable[key] += self.learning_rate*(reward - self.qtable[key])

        else: # 게임이 진행중일 경우 학습
            available_action = env.get_action()
            qvalues = np.zeros(len(available_action))

            for i, act in enumerate(available_action):
                next_key = (tuple(env.board_a),act)
                # 다음 상태를 경험한 적이 없다면(딕셔너리에 없다면) 딕셔너리에 추가(Q-value = 0)
                if self.qtable.get(next_key) ==  None:
                    self.qtable[next_key] = 0
                qvalues[i] = self.qtable.get(next_key)

            # maxQ 조사
            maxQ = np.max(qvalues)

            self.qtable[key] += self.learning_rate*(reward + self.gamma * maxQ - self.qtable[key])
