# Fruit Ninja Game

This is a simple Fruit Ninja game implemented using Python, OpenCV, and MediaPipe. The game uses hand tracking to allow the player to slice fruits appearing on the screen.

## Features

- Hand tracking using MediaPipe
- Randomly spawning fruits
- Different difficulty levels
- Score and lives tracking
- Game over screen

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Fruitninja-Game
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe numpy
    ```

## Usage

1. Run the game:
    ```sh
    python game.py
    ```

2. To check the camera setup:
    ```sh
    python check.py
    ```

3. To run the game with fruit images:
    ```sh
    python update.py
    ```

## Game Controls

- Use your hand to slice the fruits appearing on the screen.
- The game will track your index finger to detect slicing motion.
- The game ends when you run out of lives.

## File Structure

- `game.py`: Main game file with basic fruit spawning and hand tracking.
- `check.py`: Script to check the camera setup and hand tracking.
- `update.py`: Enhanced game file with fruit images and improved mechanics.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)