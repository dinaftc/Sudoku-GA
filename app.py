from flask import Flask, request, jsonify, render_template
from sudoku_solver import solve
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    """Render the main Sudoku solver page."""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    """Solve the Sudoku puzzle."""
    data = request.json
    puzzle = data.get('puzzle')

    # Log the received puzzle
    logging.debug(f"Received puzzle: {puzzle}")

    # Validate input
    if not puzzle or len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
        return jsonify({'error': 'Invalid Sudoku grid'}), 400

    try:
        # Solve the puzzle
        solution = solve(puzzle)

        if solution is not None:
            logging.debug("Solved Sudoku successfully.")
            return jsonify({'solution': solution})
        else:
            logging.error("No solution found.")
            return jsonify({'error': 'No solution found'}), 500

    except Exception as e:
        logging.exception("An error occurred while solving the Sudoku puzzle.")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
