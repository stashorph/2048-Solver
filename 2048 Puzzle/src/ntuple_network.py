import math
import copy
from game2048 import Game2048

class NTupleNetwork:
    """
    The network uses a set of patterns (N-tuples) on the
    board. It calculates a score based on the weights associated with
    the patterns, along with some evaluations.
    """
    def __init__(self, board_size=4, max_tile_power=16):
        self.board_size = board_size
        self.max_power = max_tile_power
        self.tuples = self._create_tuples()
        self.weights = self._initialize_weights()

    def _create_tuples(self):
        """
        Defines the N-tuples used as features.
        Creates a list of tuples that represent patterns on the board.
        """
        tuples = []
        # Horizontal rows
        for i in range(self.board_size):
            tuples.append([(i, j) for j in range(self.board_size)])

        # Vertical columns
        for j in range(self.board_size):
            tuples.append([(i, j) for i in range(self.board_size)])

        # 2x2 squares
        for i in range(self.board_size - 1):
            for j in range(self.board_size - 1):
                tuples.append([(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)])
        
        return tuples

    def _initialize_weights(self):
        """Initializes weight tables for each N-tuple with empty dictionaries."""
        return [{} for _ in self.tuples]

    def get_pattern_index(self, board, tuple_coords):
        """
        Calculates a unique index for a given pattern of tiles on the board.

        This works by converting the sequence of tile powers in a tuple to a
        single integer, treating it as a number in a different base.
        """
        pattern = 0
        for i, (row, col) in enumerate(tuple_coords):
            tile_value = board[row][col]
            power = int(math.log2(tile_value)) if tile_value > 0 else 0
            pattern += power * (self.max_power ** i)
        return pattern

    def evaluate(self, board):
        """
        Evaluates the given board state by summing N-tuple weights and heuristics.
        """
        total_value = 0.0

        # 1. N-tuple learned weights
        for t_id, t_coords in enumerate(self.tuples):
            pattern = self.get_pattern_index(board, t_coords)
            total_value += self.weights[t_id].get(pattern, 0.0)

        # 2. Empty cells are valuable
        empty_cells = sum(row.count(0) for row in board)
        total_value += empty_cells * 50

        # 3. Monotonicity (encourages ordered rows/columns)
        monotonicity_score = 0
        for i in range(self.board_size):
            row_tiles = [value for value in board[i] if value > 0]
            if len(row_tiles) > 1:
                # Check if tiles are sorted in increasing/decreasing order
                is_increasing = all(row_tiles[j] <= row_tiles[j+1] for j in range(len(row_tiles) - 1))
                is_decreasing = all(row_tiles[j] >= row_tiles[j+1] for j in range(len(row_tiles) - 1))

                if is_increasing or is_decreasing:
                    monotonicity_score += 50

            # Get a list of only the non-zero tiles in the column
            column = [board[j][i] for j in range(self.board_size)]
            col_tiles = [value for value in column if value > 0]

            if len(col_tiles) > 1:
                # Check for increasing/decreasing order
                is_increasing = all(col_tiles[j] <= col_tiles[j+1] for j in range(len(col_tiles) - 1))
                is_decreasing = all(col_tiles[j] >= col_tiles[j+1] for j in range(len(col_tiles) - 1))

                if is_increasing or is_decreasing:
                    monotonicity_score += 50

        total_value += monotonicity_score

        # 4. High tiles in corners are good
        corners = [(0, 0), (0, self.board_size-1), (self.board_size-1, 0), (self.board_size-1, self.board_size-1)]
        highest_tile = max(max(row) for row in board)
        if highest_tile >= 64:
            for r, c in corners:
                if board[r][c] == highest_tile:
                    total_value += highest_tile
                    break # Only reward one corner

        return total_value


class NTupleSolver:
    """
    Uses an NTupleNetwork to find the best move in a 2048 game.
    """
    def __init__(self, game: Game2048, network: NTupleNetwork):
        self.game = game
        self.network = network

    def get_best_move(self):
        """
        Determines the best move by evaluating the outcome of each possible action.
        """
        if self.game.game_over:
            return None

        best_move = None
        best_value = float('-inf')

        for move in ["up", "down", "left", "right"]:
            game_copy = copy.deepcopy(self.game)
            if game_copy.make_move(move):
                # The state after the move but before a new tile is added
                # is often called the "afterstate".
                value = self.network.evaluate(game_copy.board)
                if value > best_value:
                    best_value = value
                    best_move = move
        
        return best_move

    def make_move(self):
        """Finds and executes the best move on the game board."""
        best_move = self.get_best_move()
        if best_move:
            self.game.make_move(best_move)
            return True
        return False

