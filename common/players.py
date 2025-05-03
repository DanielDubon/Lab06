from common.game import TicTacToe

class MinimaxPlayer:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.nodes_explored = 0

    def get_move(self, game: TicTacToe):
        self.nodes_explored = 0
        best_score = float('-inf')
        best_move = None
        for move in game.available_moves():
            new_game = game.clone()
            new_game.make_move(move)
            score = self.minimax(new_game, 0, False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, self.nodes_explored

    def minimax(self, game: TicTacToe, depth: int, is_maximizing: bool):
        self.nodes_explored += 1
        if game.game_over:
            return game.get_utility()
        if depth >= self.max_depth:
            return game.evaluate_heuristic()
        if is_maximizing:
            best_score = float('-inf')
            for move in game.available_moves():
                next_game = game.clone()
                next_game.make_move(move)
                score = self.minimax(next_game, depth + 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in game.available_moves():
                next_game = game.clone()
                next_game.make_move(move)
                score = self.minimax(next_game, depth + 1, True)
                best_score = min(best_score, score)
            return best_score

class AlphaBetaPlayer:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.nodes_explored = 0

    def get_move(self, game: TicTacToe):
        self.nodes_explored = 0
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        for move in game.available_moves():
            next_game = game.clone()
            next_game.make_move(move)
            score = self.alpha_beta(next_game, 0, False, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move, self.nodes_explored

    def alpha_beta(self, game: TicTacToe, depth: int, is_maximizing: bool, alpha: float, beta: float):
        self.nodes_explored += 1
        if game.game_over:
            return game.get_utility()
        if depth >= self.max_depth:
            return game.evaluate_heuristic()
        if is_maximizing:
            best_score = float('-inf')
            for move in game.available_moves():
                next_game = game.clone()
                next_game.make_move(move)
                score = self.alpha_beta(next_game, depth + 1, False, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in game.available_moves():
                next_game = game.clone()
                next_game.make_move(move)
                score = self.alpha_beta(next_game, depth + 1, True, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
