import numpy as np
from tqdm import tqdm

import mnk_env, mnk_dqn, mnk_qt

np.random.seed(0)

p1_DQN = mnk_dqn.DQN_player()
p2 = mnk_qt.Q_learning_player()
p2.epsilon = 0.5

p1_score = 0
p2_score = 0
draw_score = 0

max_learn = 100

train_history = []

for j in tqdm(range(max_learn)):
    np.random.seed(j)
    env = mnk_env.Environment()

    # 시작할 때 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
    p1_DQN.epsilon = 0.7
    p1_DQN.copy_network()

    for i in range(10000):
        # p1 행동을 선택
        player = 1
        pos = p1_DQN.policy(env)

        p1_board_backup = tuple(env.board_a)
        p1_action_backup = pos

        env.board_a[pos] = player
        env.end_check(player)

        # 게임 종료라면
        if env.done == True:
            # p1의 승리이므로 마지막 행동에 보상 +1
            # p2는 마지막 행동에 보상 -1
            # p1 행동의 결과는 이기거나 비기거나
            if env.reward == 0:
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)
                draw_score += 1
                break
            else:
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 1)
                p1_score += 1
                break

        # p2 행동을 선택
        player = -1
        pos = p2.select_action(env)
        env.board_a[pos] = player
        env.end_check(player)

        if env.done == True:
            # p2승리 = p1 패배 마지막 행동에 보상 -1
            # 비기면 보상 : 0
            if env.reward == 0:
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)
                draw_score += 1
                break
            else:
                # 지면 보상 : -1
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, -1)
                p2_score += 1
                break
        # 게임이 끝나지 않았다면 p1의 Q-talble 학습
        p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)

    # 5게임마다 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
    if j%5 == 0:
        p1_DQN.copy_network()

    train_history.append([j, p1_score, p2_score, draw_score])

print("p1 = {} p2 = {} draw = {}".format(p1_score,p2_score,draw_score))
print("end learn")

p1_DQN.save_network("p1_DQN")
