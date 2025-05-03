import numpy as np
import time
import random

class TicTacToe:
    def __init__(self, first_player='X'):
        # Inicializar un tablero vacío de 3x3
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = first_player  # 'X' o 'O'
        self.winner = None
        self.game_over = False
        
    def reset(self, first_player='X'):
        # Reiniciar el juego
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = first_player
        self.winner = None
        self.game_over = False
        
    def available_moves(self):
        # Devolver lista de posiciones disponibles como tuplas (fila, columna)
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def make_move(self, position):
        # Realizar un movimiento en la posición dada
        if not self.game_over and self.board[position[0]][position[1]] == ' ':
            self.board[position[0]][position[1]] = self.current_player
            self.check_winner()
            # Cambiar al siguiente jugador
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def check_winner(self):
        # Verificar filas, columnas y diagonales para un ganador
        # Verificar filas
        for i in range(3):
            if self.board[i][0] != ' ' and self.board[i][0] == self.board[i][1] == self.board[i][2]:
                self.winner = self.board[i][0]
                self.game_over = True
                return
        
        # Verificar columnas
        for i in range(3):
            if self.board[0][i] != ' ' and self.board[0][i] == self.board[1][i] == self.board[2][i]:
                self.winner = self.board[0][i]
                self.game_over = True
                return
        
        # Verificar diagonales
        if self.board[0][0] != ' ' and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            self.winner = self.board[0][0]
            self.game_over = True
            return
        
        if self.board[0][2] != ' ' and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            self.winner = self.board[0][2]
            self.game_over = True
            return
        
        # Verificar empate
        if len(self.available_moves()) == 0:
            self.game_over = True
            return
    
    def get_utility(self):
        # Devolver el valor de utilidad del estado actual
        if self.winner == 'X':
            return 1  # Victoria para X
        elif self.winner == 'O':
            return -1  # Victoria para O
        elif self.game_over:
            return 0  # Empate
        else:
            return None  # Juego no terminado
    
    def evaluate_heuristic(self):
        """
        Función de evaluación heurística para un estado no terminal.
        Valores positivos favorecen a 'X', valores negativos favorecen a 'O'.
        """
        if self.winner == 'X':
            return 1
        elif self.winner == 'O':
            return -1
        elif self.game_over:
            return 0
        
        # Heurística: contar líneas con potenciales victorias
        score = 0
        
        # Verificar filas
        for i in range(3):
            row = self.board[i]
            if ' ' in row and 'X' in row and 'O' not in row:
                score += 0.1 * row.tolist().count('X')
            if ' ' in row and 'O' in row and 'X' not in row:
                score -= 0.1 * row.tolist().count('O')
        
        # Verificar columnas
        for i in range(3):
            col = self.board[:, i]
            if ' ' in col and 'X' in col and 'O' not in col:
                score += 0.1 * col.tolist().count('X')
            if ' ' in col and 'O' in col and 'X' not in col:
                score -= 0.1 * col.tolist().count('O')
        
        # Verificar diagonales
        diag1 = [self.board[i][i] for i in range(3)]
        if ' ' in diag1 and 'X' in diag1 and 'O' not in diag1:
            score += 0.1 * diag1.count('X')
        if ' ' in diag1 and 'O' in diag1 and 'X' not in diag1:
            score -= 0.1 * diag1.count('O')
        
        diag2 = [self.board[i][2-i] for i in range(3)]
        if ' ' in diag2 and 'X' in diag2 and 'O' not in diag2:
            score += 0.1 * diag2.count('X')
        if ' ' in diag2 and 'O' in diag2 and 'X' not in diag2:
            score -= 0.1 * diag2.count('O')
        
        # La posición central tiene ventaja estratégica
        if self.board[1][1] == 'X':
            score += 0.2
        elif self.board[1][1] == 'O':
            score -= 0.2
            
        return score
    
    def print_board(self):
        # Imprimir el estado actual del tablero
        for i in range(3):
            print('|'.join(self.board[i]))
            if i < 2:
                print('-' * 5)
    
    def clone(self):
        # Crear una copia del estado actual del juego
        clone = TicTacToe(self.current_player)
        clone.board = np.copy(self.board)
        clone.winner = self.winner
        clone.game_over = self.game_over
        return clone

# Implementación del algoritmo Minimax con poda Alpha-Beta
class AlphaBetaPlayer:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.nodes_explored = 0
    
    def get_move(self, game):
        self.nodes_explored = 0
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for move in game.available_moves():
            new_game = game.clone()
            new_game.make_move(move)
            score = self.alpha_beta(new_game, 0, False, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            # Actualizar alpha
            alpha = max(alpha, best_score)
                
        return best_move, self.nodes_explored
    
    def alpha_beta(self, game, depth, is_maximizing, alpha, beta):
        self.nodes_explored += 1
        
        # Condiciones terminales
        if game.game_over:
            return game.get_utility()
        
        if depth >= self.max_depth:
            return game.evaluate_heuristic()
        
        if is_maximizing:
            best_score = float('-inf')
            for move in game.available_moves():
                new_game = game.clone()
                new_game.make_move(move)
                score = self.alpha_beta(new_game, depth + 1, False, alpha, beta)
                best_score = max(score, best_score)
                
                # Actualizar alpha y verificar poda beta
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Poda beta (corte)
                    
            return best_score
        else:
            best_score = float('inf')
            for move in game.available_moves():
                new_game = game.clone()
                new_game.make_move(move)
                score = self.alpha_beta(new_game, depth + 1, True, alpha, beta)
                best_score = min(score, best_score)
                
                # Actualizar beta y verificar poda alpha
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Poda alpha (corte)
                    
            return best_score

# Función para ejecutar los experimentos
def run_experiments(first_player='X', num_trials=1000, max_depth=3):
    # Variables para almacenar resultados
    wins = 0
    losses = 0
    draws = 0
    total_nodes = 0
    total_time = 0
    
    alpha_beta_player = AlphaBetaPlayer(max_depth=max_depth)
    
    print(f"Ejecutando {num_trials} experimentos con Alpha-Beta (profundidad={max_depth})...")
    
    for trial in range(num_trials):
        game = TicTacToe(first_player)
        
        while not game.game_over:
            if game.current_player == 'X':
                # El jugador X usa Alpha-Beta
                start_time = time.time()
                move, nodes = alpha_beta_player.get_move(game)
                end_time = time.time()
                
                total_nodes += nodes
                total_time += (end_time - start_time)
                
                game.make_move(move)
            else:
                # El jugador O hace movimientos aleatorios
                moves = game.available_moves()
                if moves:
                    move = random.choice(moves)
                    game.make_move(move)
        
        # Registrar el resultado
        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1
            
        if (trial + 1) % 100 == 0:
            print(f"  Progreso: {trial + 1}/{num_trials} pruebas completadas")
    
    # Calcular promedios
    avg_nodes = total_nodes / num_trials
    avg_time = total_time / num_trials
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'avg_nodes': avg_nodes,
        'avg_time': avg_time
    }

# Función para mostrar los resultados
def display_results(results, first_player, depth):
    win_rate = results['wins'] / (results['wins'] + results['losses'] + results['draws']) * 100
    print(f"\nResultados con Alpha-Beta (profundidad={depth}) cuando {first_player} juega primero:")
    print(f"Victorias: {results['wins']} ({win_rate:.2f}%)")
    print(f"Derrotas: {results['losses']}")
    print(f"Empates: {results['draws']}")
    print(f"Nodos explorados promedio: {results['avg_nodes']:.2f}")
    print(f"Tiempo promedio por movimiento: {results['avg_time']:.6f} segundos")

# Función principal para ejecutar el experimento
def main():
    # Configuración del experimento
    max_depth = 3  # Profundidad de búsqueda
    num_trials = 1000  # Número de pruebas
    
    # Experimentos con 'X' (Alpha-Beta) jugando primero
    print("Ejecutando experimentos con 'X' (Alpha-Beta) jugando primero...")
    results_x_first = run_experiments(first_player='X', num_trials=num_trials, max_depth=max_depth)
    
    # Experimentos con 'O' (oponente) jugando primero
    print("\nEjecutando experimentos con 'O' (oponente) jugando primero...")
    results_o_first = run_experiments(first_player='O', num_trials=num_trials, max_depth=max_depth)
    
    # Mostrar los resultados
    display_results(results_x_first, 'X', max_depth)
    display_results(results_o_first, 'O', max_depth)

    # Comparar resultados con diferentes profundidades
    print("\nComparando rendimiento con diferentes profundidades de búsqueda:")
    for depth in [1, 2, 4, 5]:
        print(f"\nProfundidad de búsqueda: {depth}")
        
        # X juega primero
        results = run_experiments(first_player='X', num_trials=100, max_depth=depth)
        display_results(results, 'X', depth)
        
        # O juega primero
        results = run_experiments(first_player='O', num_trials=100, max_depth=depth)
        display_results(results, 'O', depth)

if __name__ == "__main__":
    main()