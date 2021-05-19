import numpy as np
import pickle

class boardstate:
    def __init__(self,p1,p2):
        self.board = np.zeros((3,3))
        self.gameover = False
        self.boardHash = None
        self.p1 = p1
        self.p2 = p2

    def hashfn(self):
        return str(self.board.reshape(9))

    def reset(self):
        self.board = np.zeros((3,3))
        self.gameover = False

    def availPos(self):
        pos = []
        for i in range(3):
            for j in range(3):
                if self.board[i,j] == 0:
                    pos.append((i,j))
        return pos

    def winner(self):
        for i in range(3):
            if sum(self.board[i, :]) == 3:
                self.gameover = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.gameover = True
                return -1
            if sum(self.board[:, i]) == 3:
                self.gameover = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.gameover = True
                return -1
        if sum(self.board[i,i] for i in range (3)) == 3 or sum(self.board[i, 2 - i] for i in range(3)) == 3:
            self.gameover = True
            return 1
        if sum(self.board[i,i] for i in range (3)) == -3 or sum(self.board[i, 2 - i] for i in range(3)) == -3:
            self.gameover = True
            return -1

        if len(self.availPos()) == 0:
            self.gameover = True
            return 0
        return None

    def reward(self,winner):
        if winner == 1:
            self.p1.givereward(1)
            self.p2.givereward(0)
        elif winner == -1:
            self.p1.givereward(0)
            self.p2.givereward(1)
        else:
            self.p1.givereward(0.5)
            self.p2.givereward(0.5)


    def update_board(self,action,num):
        self.board[action] = num

    def print_board(self):
        for i in range(3):
            if i != 0: print("-----")
            output = ""
            for j in range(3):
                if j > 0: output += "|"
                if self.board[i,j] == 1: output += "X"
                elif self.board[i,j] == -1: output += "O"
                else: output += " "
            print(output)
        return

    def cpu_cpu(self,rounds):
        for i in range(rounds):
            if i % 5000 == 0: print("round " + str(i))
            while not self.gameover:
                #player 1
                pos = self.availPos()
                action = self.p1.action(pos, self.board, 1)
                self.update_board(action,1)
                self.boardHash = self.hashfn()
                self.p1.add_state(self.boardHash)
                win = self.winner()
                if win is not None:
                    self.reward(win)
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    #player 2
                    pos = self.availPos()
                    action = self.p2.action(pos, self.board, -1)
                    self.update_board(action, -1)
                    self.boardHash = self.hashfn()
                    self.p2.add_state(self.boardHash)
                    win = self.winner()
                    if win is not None:
                        self.reward(win)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    def cpu_human(self):
        while not self.gameover:
            #player 1
            pos = self.availPos()
            action = self.p1.action(pos, self.board, 1)
            self.update_board(action,1)
            self.print_board()
            win = self.winner()
            if win is not None:
                if win == 1: print("cpu win")
                else: print("tie")
                self.reset()
                break

            else:
                #player 2
                pos = self.availPos()
                action = self.p2.action(pos)
                self.update_board(action, -1)
                win = self.winner()
                if win is not None:
                    if win == -1: print("u win")
                    else: print("tie")
                    self.reset()
                    break

class player:
    def __init__(self,name, exp_rate=0):
        self.name = name
        self.explore_rate = exp_rate
        self.states = []
        self.state_qvals = {} #this is the dictionary of states to reward value mappings

    def hashfn(self, board):
        return str(board.reshape(9))

    #exploration - 0.5, exploitation - 0.5
    #@param[in] pos Array of the possible moves to take
    #@param[in] curr_board the current board
    #@param[in] player the player who is making the move
    def action(self, pos, curr_board, player):
        rng = np.random.normal(0,1,1)
        #exploration
        if rng < self.explore_rate:
            index = np.random.choice(len(pos))
            move = pos[index]
        #exploitation -- find highest reward for each of your moves
        else:
            max_value = -1e9
            for i in pos:
                next_board = curr_board.copy()
                next_board[i] = player
                next_boardHash = self.hashfn(next_board)
                qval = 0 if self.state_qvals.get(next_boardHash) is None else self.state_qvals.get(next_boardHash)
                if qval > max_value:
                    max_value = qval
                    move = i
        return move

    def add_state(self,state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def givereward(self,amount):
        for state in reversed(self.states):
            if self.state_qvals.get(state) is None:
                self.state_qvals[state] = 0
            self.state_qvals[state] += 0.1 * (amount - self.state_qvals[state])
            amount = self.state_qvals[state]

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.state_qvals, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.state_qvals = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def action(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row-1, col-1)
            if action in positions:
                return action
            else: print("invalid action")

if __name__ == "__main__":
    #p1 = player("p1", exp_rate=0.5)
    #p2 = player("p2", exp_rate=0.5)

    #board = boardstate(p1,p2)
    #board.cpu_cpu(50000)

    #p1.savePolicy()
    #p2.savePolicy()

    p1 = player("computer")
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = boardstate(p1, p2)
    st.cpu_human()