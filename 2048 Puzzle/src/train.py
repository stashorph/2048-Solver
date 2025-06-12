import pickle
import time
import copy
import argparse
import os
import glob

from game2048 import Game2048
from ntuple_network import Ntuplenetwork

class TDLearner:
    """
    Implements temporal-difference (TD) learning to train an NTupleNetwork.

    The learner plays games of 2048, and for each move, it updates the network's
    weights based on the reward received and the value of the state.
    """
    def __init__(self, network, alpha=0.01, gamma=0.95):

        self.game = Game2048()
        self.network = network
        self.alpha = alpha
        self.gamma = gamma

    def get_reward(self, prev_score, current_score, prev_board, current_board):
        """
        Calculates a reward based on the change in game state.
        A good reward function is critical for effective training.
        """
        # 1. Reward for increasing the score (primary goal)
        score_reward = current_score - prev_score

        # 2. Bonus for creating a new highest tile
        prev_max_tile = max(max(row) for row in prev_board)
        current_max_tile = max(max(row) for row in current_board)
        tile_reward = current_max_tile if current_max_tile > prev_max_tile else 0
        
        # 3. Small bonus for keeping more cells empty
        empty_cells = sum(row.count(0) for row in current_board)
        empty_reward = empty_cells * 10

        return score_reward + tile_reward + empty_reward

    def train(self, num_episodes, save_interval, weights_dir):
        """
        Runs the main training loop for a certain number of episodes.
        """
        print(f"Training for {num_episodes} episodes...")
        start_time = time.time()
        best_tile = 0
        
        for episode in range(1, num_episodes + 1):
            self.game.reset()
            prev_afterstate = None
            prev_value = 0

            while not self.game.game_over:
                # --- Step 1: Find the best move and its state
                best_move = None
                best_afterstate = None
                best_value = float('-inf')

                for move in ["up", "down", "left", "right"]:
                    game_copy = copy.deepcopy(self.game)
                    if game_copy.make_move(move):
                        # "Afterstate" is the board after a move but before a new tile is added
                        afterstate = game_copy.board
                        value = self.network.evaluate(afterstate)
                        
                        if value > best_value:
                            best_value = value
                            best_move = move
                            best_afterstate = afterstate

                if best_move is None:
                    break 

                # Calculate reward and perform TD Update
                if prev_afterstate is not None:
                    
                    reward = self.get_reward(
                        self.game.score, self.game.score,
                        prev_afterstate, best_afterstate
                    )
                    
                    # The difference between our new estimate and the old one
                    td_error = reward + (self.gamma * best_value) - prev_value
                    
                    
                    for t_id, t_coords in enumerate(self.network.tuples):
                        pattern = self.network.get_pattern_index(prev_afterstate, t_coords)
                        current_weight = self.network.weights[t_id].get(pattern, 0.0)
                        self.network.weights[t_id][pattern] = current_weight + self.alpha * td_error
                
                # Make the best move and update the game state
                self.game.make_move(best_move)
                prev_afterstate = best_afterstate
                prev_value = best_value

            # Log progress and save weights
            current_max_tile = max(max(row) for row in self.game.board)
            if current_max_tile > best_tile:
                best_tile = current_max_tile
                print(f"  New best tile: {best_tile} at episode {episode}")

            if episode % save_interval == 0:
                self._save_weights(f"{weights_dir}/ntuple_weights_{episode}.pkl")
                elapsed = time.time() - start_time
                print(f"Episode {episode}/{num_episodes} | Time: {elapsed:.1f}s | "
                      f"Score: {self.game.score} | Highest Tile: {current_max_tile}")
        
        # Save the final weights after all trianing episodes
        self._save_weights(f"{weights_dir}/ntuple_weights_final.pkl")
        print(f"\nTraining complete in {time.time() - start_time:.1f} seconds.")
        print(f"Best tile achieved during training: {best_tile}")

    def _save_weights(self, filename):
        """Saves the network weights to a file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.network.weights, f)
        print(f"Weights saved to {filename}")

def find_latest_checkpoint(weights_dir):
    """Finds the most recent weight checkpoint file."""
    
    checkpoints = glob.glob(f"{weights_dir}/ntuple_weights_*.pkl")
    if not checkpoints:
        return None, 0

    latest_episode = -1
    latest_checkpoint = None
    for cp in checkpoints:
        try:
            # Exclude 'final' from resume logic
            if 'final' in os.path.basename(cp):
                continue
            episode_num = int(os.path.basename(cp).split('_')[2].split('.')[0])
            if episode_num > latest_episode:
                latest_episode = episode_num
                latest_checkpoint = cp
        except (IndexError, ValueError):
            continue
    
    return latest_checkpoint, latest_episode

def run_training(args):
    """Sets up and runs the training session based on command-line arguments."""
    network = Ntuplenetwork()
    
    if args.resume:
        latest_checkpoint, resume_episode = find_latest_checkpoint(args.weights_dir)
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            with open(latest_checkpoint, 'rb') as f:
                network.weights = pickle.load(f)
            start_episode = resume_episode
        else:
            print("No checkpoints found. Starting new training session.")
            start_episode = 0
    else:
        print("Starting new training session.")
        start_episode = 0

    remaining_episodes = args.episodes - start_episode
    if remaining_episodes <= 0:
        print("Training already completed for the specified number of episodes.")
        return

    trainer = TDLearner(network, alpha=args.alpha)
    trainer.train(
        num_episodes=remaining_episodes, 
        save_interval=args.save_interval, 
        weights_dir=args.weights_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an N-tuple network for 2048.")
    parser.add_argument('--episodes', type=int, default=100000, help='Total number of episodes to train for.')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save weights every N episodes.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate for the TD learner.')
    parser.add_argument('--weights-dir', type=str, default='weights', help='Directory to save/load weight files.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')
    
    args = parser.parse_args()
    run_training(args)
