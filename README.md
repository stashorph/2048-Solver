# 2048-Solver

This is my personal implementation of AI that learns to play the game 2048. The project features a 2048 game built with pygame and an AI solver that uses an N-Tuple network trained using Temporal-Difference (TD) learning.

## Features

- A 2048 game.
- An AI solver that uses an N-Tuple network to evaluate board states.
- A training script to teach the AI from scratch using reinforcement learning.
- Ability to toggle between human and AI play.

## Installation

To get this project running on your local machine, follow these steps.

### 1. Clone the repo and setup environment

```bash
git clone https://github.com/stashorph/2048-Solver.git
cd 2048-Solver

python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install pygame
```

## How to Use

### Playing the Game

To play the game or run the trained model, run `play.py` script.

**Controls:**

- **Arrow Keys:** Make a move (when AI is off).
- **`A`:** Toggle the AI on or off.
- **`R`:** Reset the game board.

The game will load the trained weights from `weights/ntuple_weights_final.pkl`.

## Project Structure

The project is organized in the `src/` directory

```
2048-Solver/
├── src/
│ ├── __init__.py
│ ├── game2048.py       # Core game logic and board mechanics
│ ├── ntuple_network.py # N-Tuple network and AI solver classes
│ ├── train.py          # Script for training the AI model
│ └── play.py           # GUI script to play the game (human or AI)
│
├── weights/
│ └── ntuple_weights_final.pkl # The final trained AI weights
│
└── README.md
```

## License

This project is licensed under the MIT License.
