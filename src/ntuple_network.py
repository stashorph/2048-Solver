from game2048 import Game
import math
import copy

class N_Tuple:
    def __init__(self):
        self.size = 4
        self.max_power = 16
        self.tuples = self.create_tuples()
        self.weights = self.initialize_weights()

    def create_tuples(self): # Defines the geometric patterns to look for.
        tuples = []
        for i in range(self.size):
            tuples.append([(i, j) for j in range(self.size)])
            tuples.append([(j, i) for j in range(self.size)])
    
        for i in range(self.size - 1):
            for j in range(self.size - 1):
                tuples.append([(i, j), (i, j+1), (i+1, j), (i+1, j+1)])
        return tuples

    def initialize_weights(self): # One empty weight table per tuple pattern.
        return [{} for _ in self.tuples]

    def get_pat_idx(self, board, tuple_coords): # Converts a board state into a unique pattern index.
        pattern = 0
        for i, (r, c) in enumerate(tuple_coords):
            tile_val = board[r][c]
            power = int(math.log2(tile_val)) if tile_val > 0 else 0
            pattern += power * (self.max_power ** i)
        return pattern

    def evaluate(self, board):
        val = 0.0

        # Ntuple learned weights
        for t_idx, t_coords in enumerate(self.tuples):
            pattern_idx = self.get_pat_idx(board, t_coords)
            val += self.weights[t_idx].get(pattern_idx, 0.0)

        # More no. of empty cells are good
        val += sum(row.count(0) for row in board) * 50

        # Monotonicity is a bonus (Rows/cols in order)
        mono_score = 0
        for i in range(self.size):
            row = [v for v in board[i] if v > 0]
            col = [board[j][i] for j in range(self.size)]
            col = [v for v in col if v > 0]
            
            if len(row) > 1 and (all(row[j] <= row[j+1] for j in range(len(row)-1)) or \
                                 all(row[j] >= row[j+1] for j in range(len(row)-1))):
                mono_score += 50
            if len(col) > 1 and (all(col[j] <= col[j+1] for j in range(len(col)-1)) or \
                                 all(col[j] >= col[j+1] for j in range(len(col)-1))):
                mono_score += 50
        val += mono_score

        # Higher tiles in corners are good
        max_tile = max(max(row) for row in board)
        if max_tile >= 64:
            corners = [(0, 0), (0, self.size-1), (self.size-1, 0), (self.size-1, self.size-1)]
            for r, c in corners:
                if board[r][c] == max_tile:
                    val += max_tile
                    break
        
        return val


class Solver:

    def __init__(self, game= Game, network= N_Tuple):
        self.game = game
        self.network = network

    def find_best_move(self): # Looks one step ahead to find the highest-scoring move.
        if self.game.over:
            return None

        best_move = None
        best_val = -float('inf')

        for move in ["up", "down", "left", "right"]:
            game_copy = copy.deepcopy(self.game)
            if game_copy.make_move(move):
                val = self.network.evaluate(game_copy.board)
                if val > best_val:
                    best_val = val
                    best_move = move
        
        return best_move

    def make_move(self): # Plays the best move on the board.
        best_move = self.find_best_move()
        if best_move:
            self.game.make_move(best_move)
            return True
        return False
