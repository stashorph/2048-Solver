import random
import copy

class Game:
    "Core 2048 game logic: board, moves, score, and game state."

    def __init__(self, size=4):
        self.size = size
        self.reset()

    def add_tile(self):
        "Adds a new 2 or 4 tile to an empty spot on the board."
        empty_spots = [(r, c) for r in range(self.size) for c in range(self.size) if not self.board[r][c]]
        
        if empty_spots:
            r, c = random.choice(empty_spots)
            # 90% chance for a 2, 10% for a 4
            self.board[r][c] = 2 if random.random() < 0.9 else 4

    def merge_line(self, line):
        "Helper function to merge one line (a row or column) to the left."
        # Shift all non-zero tiles to the left
        non_zeros = [val for val in line if val != 0]
        
        score_gain = 0
        merged_line = []
        
        i = 0
        while i < len(non_zeros):
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i+1]:
                # Merge tiles
                merged_val = non_zeros[i] * 2
                merged_line.append(merged_val)
                score_gain += merged_val
                i += 2
            else:
                merged_line.append(non_zeros[i])
                i += 1
                
        # Fill the rest of the line with zeros
        merged_line.extend([0] * (self.size - len(merged_line)))
        return merged_line, score_gain
    
    def transpose(self, board):
        return [list(row) for row in zip(*board)]

    def reverse(self, board):
        return [row[::-1] for row in board]

    def make_move(self, direction):
        if self.over:
            return False

        original_board = copy.deepcopy(self.board)
        
        # Transform board so we only need to handle a "left" merge
        temp_board = copy.deepcopy(self.board)
        if direction == 'up':
            temp_board = self.transpose(temp_board)
        elif direction == 'right':
            temp_board = self.reverse(temp_board)
        elif direction == 'down':
            temp_board = self.reverse(self.transpose(temp_board))

        # Merge all lines and calculate score
        score_gain = 0
        new_board_lines = []
        for line in temp_board:
            merged_line, gain = self.merge_line(line)
            new_board_lines.append(merged_line)
            score_gain += gain
        
        self.score += score_gain
        temp_board = new_board_lines

        # Transform board back to original orientation
        if direction == 'up':
            temp_board = self.transpose(temp_board)
        elif direction == 'right':
            temp_board = self.reverse(temp_board)
        elif direction == 'down':
            temp_board = self.transpose(self.reverse(temp_board))

        self.board = temp_board
        
        # Check if anything changed
        if self.board != original_board:
            self.add_tile()
            if not self.valid_moves():
                self.over = True
            return True

        return False

    def valid_moves(self):
        
        # Check for empty spots
        if any(0 in row for row in self.board):
            return True

        # Check for horizontal/vertical merges
        for r in range(self.size):
            for c in range(self.size):
                if c + 1 < self.size and self.board[r][c] == self.board[r][c+1]:
                    return True
                if r + 1 < self.size and self.board[r][c] == self.board[r+1][c]:
                    return True
        return False

    def reset(self): # Resets the game to its initial state.
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.over = False
        self.add_tile()
        self.add_tile()