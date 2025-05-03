from common.game import TicTacToe, display_results
import numpy as np
import random
import time

#python -m EJ_3.main

class MCTSNode:
    def __init__(self, state: TicTacToe, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = state.available_moves()
        self.visits = 0
        self.total_value = 0.0

    def uct_select_child(self, c_param: float):
        # Selecciona el hijo con el mayor valor UCT
        uct_values = [
            (child.total_value / child.visits) +
            c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[int(np.argmax(uct_values))]

    def add_child(self, move, state: TicTacToe):
        # Expande un nodo para el movimiento dado
        child = MCTSNode(state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, value: float):
        # Actualiza estadísticas tras simulación
        self.visits += 1
        self.total_value += value

class MCTSPlayer:
    def __init__(self, num_simulations=100, c_param=1.4):
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.nodes_explored = 0

    def get_move(self, game: TicTacToe):
        self.nodes_explored = 0
        root = MCTSNode(game.clone())

        for _ in range(self.num_simulations):
            node = root
            state = game.clone()

            # SELECTION
            while not state.game_over and not node.untried_moves and node.children:
                node = node.uct_select_child(self.c_param)
                state.make_move(node.move)
                self.nodes_explored += 1

            # EXPANSION
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.make_move(move)
                node = node.add_child(move, state.clone())
                self.nodes_explored += 1

            # SIMULATION
            while not state.game_over:
                mv = random.choice(state.available_moves())
                state.make_move(mv)

            # BACKPROPAGATION
            result = state.get_utility()
            while node is not None:
                node.update(result)
                node = node.parent

        # Selecciona la jugada con más visitas
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move, self.nodes_explored

def run_experiments(first_player='X', trials=1000, sims=100):
    wins = losses = draws = 0
    total_nodes = 0
    total_time = 0.0

    mcts = MCTSPlayer(num_simulations=sims)

    print(f"Ejecutando {trials} pruebas con MCTS (simulaciones={sims}), {first_player} inicia...")
    for t in range(trials):
        game = TicTacToe(first_player)
        while not game.game_over:
            if game.current_player == 'X':
                start = time.time()
                move, nodes = mcts.get_move(game)
                end = time.time()
                total_nodes += nodes
                total_time += (end - start)
                game.make_move(move)
            else:
                game.make_move(random.choice(game.available_moves()))

        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

        if (t + 1) % 100 == 0:
            print(f"  Progreso: {t+1}/{trials}")

    avg_nodes = total_nodes / trials
    avg_time = total_time / trials

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'avg_nodes': avg_nodes,
        'avg_time': avg_time
    }

def main():
    # Experimento cuando X inicia
    res_x = run_experiments('X', trials=1000, sims=100)
    display_results(res_x, "MCTS (X inicia)")

    # Experimento cuando O inicia
    res_o = run_experiments('O', trials=1000, sims=100)
    display_results(res_o, "MCTS (O inicia)")

if __name__ == "__main__":
    main()
