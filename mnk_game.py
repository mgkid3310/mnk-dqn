import itertools
import math

class mnk_env():
    def __init__(self, m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k
        self.reset()
    
    def reset(self):
        self.board = [["."] * self.n for _ in range(self.m)]

    def count(self, dy, dx):
        cnt=0
        for delta in itertools.count(1):
            if 0 <= self.column + dx*delta < self.n and \
                0 <= self.row + dy*delta < self.m and \
                self.board[self.row + dy*delta][self.column + dx*delta] == self.player:
                cnt+=1
            else:
                return cnt


    def check_victory(self, direction):
        return self.count(*direction) + self.count(-direction[0], -direction[1]) >= self.k - 1


    def process_move(self):
        self.board[self.row][self.column] = self.player
        return any(map(self.check_victory, [(0, 1), (1, 0), (1, 1), (1, -1)]))


    def format_board(self):
        format_str = "{:>" + str(math.floor(math.log(max(self.m, self.n), 10) + 1)) + "}"
        board_to_format = [[*range(1, self.n + 1)]] + self.board
        return "\n".join(
            " ".join(map(format_str.format, [row_no] + board_to_format[row_no]))
            for row_no in range(self.m + 1)
        )

    def step(self, player, row, column):
        self.player = player
        self.row = row -1
        self.column = column -1
        victory = self.process_move()
        
        return victory

if __name__ == "__main__":
    m, n, k = map(int, input("Enter M N K (M is height, N is width, K is the length of the winning sequence)\n").split())
    env = mnk_env(m, n, k)
    print(env.format_board())
    
    players = ["O", "X"]
    for player in itertools.cycle(players):
        row, column = map(int, input(f"Player {player}, please, enter row and column\n").split())
        victory = env.step(player, row, column)
        print(env.format_board())
        if victory:
            print(f"\n\nPlayer {player} won!")
            break
        elif all(all(map('.'.__ne__, row)) for row in env.board):
            print(f"Draw!")
            break