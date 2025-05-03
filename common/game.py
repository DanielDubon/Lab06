import numpy as np

class TicTacToe:
    def __init__(self, first_player='X'):
        self.board = np.array([[' '] * 3 for _ in range(3)])
        self.current_player = first_player
        self.winner = None
        self.game_over = False

    def reset(self, first_player='X'):
        self.board[:] = ' '
        self.current_player = first_player
        self.winner = None
        self.game_over = False

    def available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i,j] == ' ']

    def make_move(self, pos):
        if not self.game_over and self.board[pos] == ' ':
            self.board[pos] = self.current_player
            self.check_winner()
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        B = self.board
        lines = [B[i,:] for i in range(3)] + [B[:,j] for j in range(3)] \
                + [[B[i,i] for i in range(3)], [B[i,2-i] for i in range(3)]]
        for line in lines:
            if line[0] != ' ' and all(line[0] == c for c in line):
                self.winner = line[0]
                self.game_over = True
                return
        if not self.available_moves():
            self.game_over = True

    def get_utility(self):
        if   self.winner == 'X': return  1
        elif self.winner == 'O': return -1
        elif self.game_over:     return  0
        else:                    return None

    def clone(self):
        copy = TicTacToe(self.current_player)
        copy.board = np.copy(self.board)
        copy.winner = self.winner
        copy.game_over = self.game_over
        return copy

def display_results(results, label):
    total = results['wins']+results['losses']+results['draws']
    win_rate = results['wins']/total*100
    print(f"\n=== {label} ===")
    print(f"Victorias: {results['wins']} ({win_rate:.1f}%)\n"
          f"Derrotas:  {results['losses']}\n"
          f"Empates:   {results['draws']}\n"
          f"Nodos/mov: {results['avg_nodes']:.1f}\n"
          f"Tiempo/mov:{results['avg_time']:.4f}s")
