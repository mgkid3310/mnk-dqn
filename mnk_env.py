import numpy as np

class Environment():

    def __init__(self):
        """
        보드는 0으로 초기화된 25개의 배열로 준비
        게임종료 : done = True
        """
        self.board_a = np.zeros(25)
        self.done = False
        self.reward = 0
        self.winner = 0
        self.print = False

    def move(self, p1, p2, player):
        """
        # 각 플레이어가 선택한 행동을 표시 하고 게임 상태(진행 또는 종료)를 판단
        # p1 = 1, p2 = -1로 정의
        # 각 플레이어는 행동을 선택하는 select_action 메서드를 가짐
        """
        if player == 1:
            pos = p1.select_action(self)
        else:
            pos = p2.select_action(self)

        self.board_a[pos] = player # 보드에 플레이어의 선택을 표시
        if self.print:
            print(player)
            self.print_board()
        self.end_check(player) # 게임이 종료상태인지 아닌지를 판단

        return  self.reward, self.done

    def get_action(self):
        """
        현재 보드 상태에서 가능한 행동(둘 수 있는 장소)을 탐색하고 리스트로 반환
        """
        observation = []
        for i in range(25):
            if self.board_a[i] == 0:
                observation.append(i)
        return observation

    def end_check(self,player):
        """
        게임이 종료(승패 또는 비김)됐는지 판단
        승패 조건은 가로, 세로, 대각선 이 -1 이나 1 로 동일할 때 
        0 1 2 3 4
        5 6 7 8 9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        """
        end_condition = (
                            (0,1,2,3),(1,2,3,4),(5,6,7,8),(6,7,8,9),(10,11,12,13),(11,12,13,14),(15,16,17,18),(16,17,18,19),(20,21,22,23),(21,22,23,24), # 가로
                            (0,6,12,17),(6,12,17,22),(1,7,13,19),(5,11,17,23), # 왼쪽 위 -> 오른쪽 아래 대각선
                            (4,8,12,16),(8,12,16,20),(3,7,11,15),(9,13,17,21), # 오른쪽 위 -> 왼쪽 아래 대각선
                            (0,5,10,15),(5,10,15,20),(1,6,11,16),(6,11,16,21),(2,7,12,17),(7,12,17,22),(3,8,13,18),(8,13,18,23),(4,9,14,19),(9,14,19,24) # 세로
                        )
        for line in end_condition:
            if self.board_a[line[0]] == self.board_a[line[1]] \
                and self.board_a[line[1]] == self.board_a[line[2]] \
                and self.board_a[line[2]] == self.board_a[line[3]] \
                and self.board_a[line[0]] != 0:
                # 종료됐다면 누가 이겼는지 표시
                self.done = True
                self.reward = player
                return
        # 비긴 상태는 더는 보드에 빈 공간이 없을때
        observation = self.get_action()
        if (len(observation)) == 0:
            self.done = True
            self.reward = 0
        return

    def print_board(self):
        """
        현재 보드의 상태를 표시 
        p1 = O, p2 = X    
        """
        print("+----+----+----+----+----+")
        for i in range(5):
            for j in range(5):
                if self.board_a[5*i+j] == 1:
                    print("|  O",end=" ")
                elif self.board_a[5*i+j] == -1:
                    print("|  X",end=" ")
                else:
                    print("|   ",end=" ")
            print("|")
            print("+----+----+----+----+----+")
