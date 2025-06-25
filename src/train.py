import pickle
import copy
import os
from game2048 import Game
from ntuple_network import N_Tuple

class Trainer:
    "Handles the TD learning  to train the NTuple network."

    def __init__(self, network):
        self.game = Game()
        self.network = network
        self.alpha = 0.01
        self.gamma =0.95          

    def get_reward(self, prev_board, current_board):
        # Reward is not based on game score but on board improvements
        
        # Big bonus for creating a new highest tile
        prev_max = max(max(row) for row in prev_board)
        curr_max = max(max(row) for row in current_board)
        tile_reward = curr_max if curr_max > prev_max else 0
        
        # Small bonus for each new empty cell
        empty_reward = sum(row.count(0) for row in current_board) * 10

        return tile_reward + empty_reward

    def run_training(self, total_episodes, save_interval):
        print(f"Starting training for {total_episodes} episodes...")
        best_tile_ever = 0
        
        for episode in range(1, total_episodes + 1):
            self.game.reset()
            prev_afterstate = None
            prev_value = 0

            while not self.game.over:   # Find the best possible move and its state after the move
                best_move, best_afterstate, best_value = None, None, -float('inf')

                for move in ["up", "down", "left", "right"]:
                    game_copy = copy.deepcopy(self.game)
                    if game_copy.make_move(move):
                        afterstate = game_copy.board
                        value = self.network.evaluate(afterstate)
                        if value > best_value:
                            best_value, best_move, best_afterstate = value, move, afterstate

                if not best_move: # No valid moves left
                    break 

                # Perform the TD update if this isn't the first move
                if prev_afterstate is not None:
                    reward = self.get_reward(prev_afterstate, best_afterstate)
                    td_error = reward + (self.gamma * best_value) - prev_value
                    
                    # Apply the error to the weights of the previous state's patterns
                    for t_idx, t_coords in enumerate(self.network.tuples):
                        pat_idx = self.network.get_pat_idx(prev_afterstate, t_coords)
                        current_weight = self.network.weights[t_idx].get(pat_idx, 0.0)
                        self.network.weights[t_idx][pat_idx] = current_weight + self.alpha * td_error
                
                
                self.game.make_move(best_move)
                prev_afterstate = best_afterstate
                prev_value = best_value

            # Logging at the end of the episode
            current_max_tile = max(max(row) for row in self.game.board)
            if current_max_tile > best_tile_ever:
                best_tile_ever = current_max_tile
                print(f"  New best tile! {best_tile_ever} (episode {episode})")

            if episode % save_interval == 0:
                self.save_weights(f"{'weights'}/ntuple_weights_{episode}.pkl")
                print(f"Episode {episode}/{total_episodes} | Score: {self.game.score} | Highest Tile: {current_max_tile}")
        
        self.save_weights(f"{'weights'}/ntuple_weights_final.pkl")
        print(f"\nTraining complete. Best tile achieved: {best_tile_ever}")

    def save_weights(self, filename): # Saves the network weights to a file.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.network.weights, f)
        print(f"Weights saved to {filename}")


if __name__ == "__main__":

    total_ep = 80000
    save_interval = 5000
    weights_file = 'weights/ntuple_weights_final.pkl'
    network = N_Tuple()

    try:
        with open(weights_file, 'rb') as f:
            network.weights = pickle.load(f)
        print(f"Loaded existing weights from {weights_file} to continue training.")
    except FileNotFoundError:
        print("No weights found. Starting new training session.")

    trainer = Trainer(network)
    trainer.run_training(
        total_episodes=total_ep,
        save_interval=save_interval
    )
