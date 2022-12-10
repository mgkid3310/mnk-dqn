class Human_player():
    def __init__(self):
        self.name = "Human player"
        
    def select_action(self, env):
        while True:
            # 가능한 행동을 조사한 후 표시
            available_action = env.get_action()
            print("possible actions = {}".format(available_action))

            # 상태 번호 표시
            print("+----+----+----+----+----+")
            print("+  0  +  1  +  2  +  3  +  4  +")
            print("+----+----+----+----+----+")
            print("+  5  +  6  +  7  +  8  +  9  +")
            print("+----+----+----+----+----+")
            print("+ 10  + 11  + 12  + 13  + 14  +")
            print("+----+----+----+----+----+")
            print("+ 15  + 16  + 17  + 18  + 19  +")
            print("+----+----+----+----+----+")
            print("+ 20  + 21  + 22  + 23  + 24  +")
            print("+----+----+----+----+----+")
                        
            action = int(input("Select action(human) : ")) # 키보드로 가능한 행동을 입력 받음
            
            if action in available_action: return action # 입력받은 행동이 가능한 행동이면 반복문을 탈출
            else: print("You selected wrong action") # 아니면 행동 입력을 반복