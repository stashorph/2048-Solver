import pygame
import pickle
from game2048 import Game
from ntuple_network import N_Tuple, Solver

class GameGUI:

    SCREEN_WIDTH, SCREEN_HEIGHT = 500, 600
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
        512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
    }

    def __init__(obj, weights_path):
        pygame.init()
        obj.screen = pygame.display.set_mode((obj.SCREEN_WIDTH, obj.SCREEN_HEIGHT))
        pygame.display.set_caption("2048 AI")

        obj.title_font = pygame.font.SysFont("Arial", 48, bold=True)
        obj.score_font = pygame.font.SysFont("Arial", 24, bold=True)
        obj.tile_font = pygame.font.SysFont("Arial", 36, bold=True)
        obj.info_font = pygame.font.SysFont("Arial", 18)

        obj.game = Game()
        network = N_Tuple()
        obj.load_weights(network, weights_path)
        obj.solver = Solver(obj.game, network)
        
        obj.ai_running = False
        obj.running = True

    def load_weights(obj, network, path):
        try:
            with open(path, 'rb') as f:
                network.weights = pickle.load(f)
            print(f"Loaded weights from: {path}")
        except FileNotFoundError:
            print(f"Warning: Weights not found.")

    def draw_tile(obj, r, c, val):
        x = 50 + c * (obj.TILE_SIZE + obj.TILE_MARGIN) + obj.TILE_MARGIN
        y = 150 + r * (obj.TILE_SIZE + obj.TILE_MARGIN) + obj.TILE_MARGIN
        
        color = obj.TILE_COLORS.get(val, obj.TILE_COLORS[2048])
        pygame.draw.rect(obj.screen, color, (x, y, obj.TILE_SIZE, obj.TILE_SIZE), border_radius=5)

        if val > 0:
            text_color = obj.COLORS['text_dark'] if val <= 4 else obj.COLORS['text_light']
            text_surface = obj.tile_font.render(str(val), True, text_color)
            text_rect = text_surface.get_rect(center=(x + obj.TILE_SIZE / 2, y + obj.TILE_SIZE / 2))
            obj.screen.blit(text_surface, text_rect)

    def draw_board(obj):
        board_rect = pygame.Rect(50, 150, obj.BOARD_SIZE, obj.BOARD_SIZE)
        pygame.draw.rect(obj.screen, obj.COLORS['grid_bg'], board_rect, border_radius=5)
        
        for r in range(obj.game.size):
            for c in range(obj.game.size):
                obj.draw_tile(r, c, obj.game.board[r][c])

    def draw_ui(obj):
        obj.screen.fill(obj.COLORS['bg'])
        obj.screen.blit(obj.title_font.render("2048", True, obj.COLORS['text_dark']), (20, 20))
        score_text = obj.score_font.render(f"SCORE: {obj.game.score}", True, obj.COLORS['text_dark'])
        obj.screen.blit(score_text, score_text.get_rect(right=obj.SCREEN_WIDTH - 20, top=35))
        
        obj.screen.blit(obj.info_font.render("Arrows: Play | R: Reset | A: Toggle AI", True, obj.COLORS['text_dark']), (20, 90))
        ai_status = "ON" if obj.ai_running else "OFF"
        ai_color = (0, 150, 0) if obj.ai_running else (200, 0, 0)
        obj.screen.blit(obj.info_font.render(f"AI: {ai_status}", True, ai_color), (20, 110))

    def draw_game_over(obj):
        overlay = pygame.Surface((obj.SCREEN_WIDTH, obj.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        obj.screen.blit(overlay, (0, 0))
        
        text = obj.title_font.render("Game Over!", True, obj.COLORS['text_dark'])
        obj.screen.blit(text, text.get_rect(center=(obj.SCREEN_WIDTH / 2, obj.SCREEN_HEIGHT / 2)))

    def run(obj): # The main game loop.
        clock = pygame.time.Clock()
        while obj.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    obj.running = False
                elif event.type == pygame.KEYDOWN:
                    # System controls
                    if event.key == pygame.K_r:
                        obj.game.reset()
                        obj.ai_running = False
                    elif event.key == pygame.K_a:
                        obj.ai_running = not obj.ai_running
                    # Player controls (if AI is off)
                    elif not obj.ai_running:
                        if event.key == pygame.K_UP: obj.game.make_move("up")
                        elif event.key == pygame.K_DOWN: obj.game.make_move("down")
                        elif event.key == pygame.K_LEFT: obj.game.make_move("left")
                        elif event.key == pygame.K_RIGHT: obj.game.make_move("right")
            
            if obj.ai_running and not obj.game.over:
                obj.solver.make_move()
                pygame.time.delay(100)
            
            obj.draw_ui()
            obj.draw_board()
            if obj.game.over:
                obj.draw_game_over()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()


gui = GameGUI(weights_path='weights/ntuple_weights_final.pkl')
gui.run()
