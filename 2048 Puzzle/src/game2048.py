import random
import copy

class Game2048:
    """
    Manages the core logic for the 2048 game.
    It handles the game board, player moves (up, down, left, right),
    tile merging, scoring, and game state.
    """
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.score = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """
        Adds a new tile (either 2 or 4) to a random empty cell on the board.
        A '2' is added 90% of the time, and a '4' is added 10% of the time.
        """
        empty_cells = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def merge_row_left(self, row):
        """
        Merges a single row to the left.
        
        Args:
            row (list): The row to merge.
            
        Returns:
            tuple: A tuple containing the new merged row (list) and the score
                   increase from merges (int).
        """
        new_row = [i for i in row if i != 0]
        score_increase = 0
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i+1]:
                merged_val = new_row[i] * 2
                new_row[i] = merged_val
                score_increase += merged_val
                new_row.pop(i + 1)
            i += 1

        #Fill rest of row with zeros
        new_row.extend([0] * (self.board_size - len(new_row)))
        return new_row, score_increase

    def transpose(self, board):
        """Transposes the game board."""
        return [list(row) for row in zip(*board)]

    def reverse(self, board):
        """Reverses each row of the game board."""
        return [row[::-1] for row in board]

    def make_move(self, direction):
        """
        Makes a move in the specified direction ('up', 'down', 'left', 'right').

        Args:
            direction (str): The direction of the move.

        Returns:
            bool: True if the board changed after the move, False otherwise.
        """
        if self.game_over:
            return False

        original_board = copy.deepcopy(self.board)
        original_score = self.score

        if direction == "left":
            new_board = []
            for i in range(self.board_size):
                new_row, score_increase = self.merge_row_left(self.board[i])
                self.score += score_increase
                new_board.append(new_row)
            self.board = new_board

        elif direction == "right":
            reversed_board = self.reverse(self.board)
            new_board = []
            for i in range(self.board_size):
                new_row, score_increase = self.merge_row_left(reversed_board[i])
                self.score += score_increase
                new_board.append(new_row)
            self.board = self.reverse(new_board)

        elif direction == "up":
            transposed_board = self.transpose(self.board)
            new_board = []
            for i in range(self.board_size):
                new_row, score_increase = self.merge_row_left(transposed_board[i])
                self.score += score_increase
                new_board.append(new_row)
            self.board = self.transpose(new_board)

        elif direction == "down":
            transposed_board = self.transpose(self.board)
            reversed_board = self.reverse(transposed_board)
            new_board = []
            for i in range(self.board_size):
                new_row, score_increase = self.merge_row_left(reversed_board[i])
                self.score += score_increase
                new_board.append(new_row)
            self.board = self.transpose(self.reverse(new_board))

        board_changed = original_board != self.board
        if board_changed:
            self.add_random_tile()

        if not self.has_valid_moves():
            self.game_over = True
            
        return board_changed

    def has_valid_moves(self):
        """  bool: True if a move is possible, False otherwise."""
        # Check for empty cells
        if any(0 in row for row in self.board):
            return True

        # Check for adjacent identical tiles (horizontally and vertically)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if j < self.board_size - 1 and self.board[i][j] == self.board[i][j+1]:
                    return True
                if i < self.board_size - 1 and self.board[i][j] == self.board[i+1][j]:
                    return True
        
        return False

    def reset(self):
        """Resets the game to its initial state."""
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.score = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()

