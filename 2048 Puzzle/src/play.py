import pygame
import pickle
import argparse
import os

from game2048 import Game2048
from ntuple_network import NTupleNetwork, NTupleSolver

class GameGUI:
    """
    Manages the GUI for the 2048 game using Pygame.
    It allows for both human play and AI play.
    """
    # Constants
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 600
    BOARD_SIZE = 395
    TILE_SIZE = 80
    TILE_MARGIN = 15

    COLORS = {
        'bg': (250, 248, 239),
        'grid_bg': (187, 173, 160),
        'text_dark': (119, 110, 101),
        'text_light': (249, 246, 242)
    }
    TILE_COLORS = {
        0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
        8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
        64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
        512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46), 
        4096: (237, 194, 46), 8192: (237, 194, 46), 16384: (237, 194, 46)
    }

    def __init__(self, weights_path):
        """
        Initializes the Pygame window, loads fonts, and sets up the game
        and AI solver.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("2048 AI Player")

        # Load fonts
        self.title_font = pygame.font.SysFont("Arial", 48, bold=True)
        self.score_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.tile_font = pygame.font.SysFont("Arial", 36, bold=True)
        self.info_font = pygame.font.SysFont("Arial", 18)

        # Initialize game and AI
        self.game = Game2048()
        self.network = NTupleNetwork()
        self.load_weights(weights_path)
        self.solver = NTupleSolver(self.game, self.network)
        
        self.ai_running = False
        self.running = True

    def load_weights(self, path):
        """Loads weights from a pickle file into the network."""
        try:
            with open(path, 'rb') as f:
                self.network.weights = pickle.load(f)
            print(f"Successfully loaded weights from: {path}")
        except FileNotFoundError:
            print(f"Warning: Weights file not found at '{path}'. AI will play with untrained heuristics.")
        except Exception as e:
            print(f"Error loading weights: {e}. AI will use untrained heuristics.")

    def draw_tile(self, row, col, value):
        """Draws a single tile on the board."""
        x = (self.SCREEN_WIDTH - self.BOARD_SIZE) / 2 + col * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_MARGIN
        y = 150 + row * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_MARGIN
        
        # Tile background
        color = self.TILE_COLORS.get(value, self.TILE_COLORS[2048])
        pygame.draw.rect(self.screen, color, (x, y, self.TILE_SIZE, self.TILE_SIZE), border_radius=5)

        # Tile text
        if value > 0:
            text_color = self.COLORS['text_dark'] if value in [2, 4] else self.COLORS['text_light']
            text_surface = self.tile_font.render(str(value), True, text_color)
            text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE / 2, y + self.TILE_SIZE / 2))
            self.screen.blit(text_surface, text_rect)

    def draw_board(self):
        """Draws the entire game board, including all tiles."""
        board_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.BOARD_SIZE) / 2, 150,
            self.BOARD_SIZE, self.BOARD_SIZE
        )
        pygame.draw.rect(self.screen, self.COLORS['grid_bg'], board_rect, border_radius=5)
        
        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                self.draw_tile(r, c, self.game.board[r][c])

    def draw_ui(self):
        """Draws static UI elements like title, score, and instructions."""
        self.screen.fill(self.COLORS['bg'])
        
        # Title
        title_text = self.title_font.render("2048", True, self.COLORS['text_dark'])
        self.screen.blit(title_text, (20, 20))

        # Score
        score_text = self.score_font.render(f"SCORE: {self.game.score}", True, self.COLORS['text_dark'])
        score_rect = score_text.get_rect(right=self.SCREEN_WIDTH - 20, top=35)
        self.screen.blit(score_text, score_rect)
        
        # Instructions
        ai_status = "ON" if self.ai_running else "OFF"
        ai_color = (0, 150, 0) if self.ai_running else (200, 0, 0)
        
        info1 = self.info_font.render("Use Arrow Keys to play.", True, self.COLORS['text_dark'])
        info2 = self.info_font.render("Press 'R' to Reset.", True, self.COLORS['text_dark'])
        info3 = self.info_font.render("Press 'A' to toggle AI.", True, self.COLORS['text_dark'])
        ai_text = self.info_font.render(f"AI Status: {ai_status}", True, ai_color)

        self.screen.blit(info1, (20, 90))
        self.screen.blit(info2, (20, 110))
        self.screen.blit(info3, (self.SCREEN_WIDTH - info3.get_width() - 20, 90))
        self.screen.blit(ai_text, (self.SCREEN_WIDTH - ai_text.get_width() - 20, 110))

    def draw_game_over(self):
        """Displays the 'Game Over' overlay."""
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.title_font.render("Game Over!", True, self.COLORS['text_dark'])
        text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(game_over_text, text_rect)
        
        restart_text = self.score_font.render("Press 'R' to play again", True, self.COLORS['text_dark'])
        restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
        self.screen.blit(restart_text, restart_rect)

    def run(self):
        """The main game loop that handles events, updates, and drawing."""
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and not self.ai_running:
                    if event.key == pygame.K_UP: self.game.make_move("up")
                    elif event.key == pygame.K_DOWN: self.game.make_move("down")
                    elif event.key == pygame.K_LEFT: self.game.make_move("left")
                    elif event.key == pygame.K_RIGHT: self.game.make_move("right")
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game.reset()
                        self.ai_running = False
                    elif event.key == pygame.K_a:
                        self.ai_running = not self.ai_running
            
            
            if self.ai_running and not self.game.game_over:
                self.solver.make_move()
                pygame.time.delay(100) # Small delay to make AI moves visible
            
            # Drawing calls
            self.draw_ui()
            self.draw_board()
            if self.game.game_over:
                self.draw_game_over()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play 2048 with an optional AI.")
    parser.add_argument(
        '--weights', 
        type=str, 
        default='2048 Puzzle/weights/final.pkl',
        help='Path to the weights file for the AI solver'
    )
    args = parser.parse_args()

    game_gui = GameGUI(weights_path=args.weights)
    game_gui.run()
