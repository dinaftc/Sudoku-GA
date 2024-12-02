import numpy as np
import random

class SudokuGA:
    def __init__(self, puzzle, population_size=500, mutation_rate=0.05, generations=1000):
        self.puzzle = np.array(puzzle)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.best_solution = None

    def is_fixed(self, row, col):
        return self.puzzle[row, col] != 0

    def fitness(self, individual):
        row_score = sum(len(set(individual[row, :])) for row in range(9))
        col_score = sum(len(set(individual[:, col])) for col in range(9))
        box_score = sum(len(set(individual[row_start:row_start + 3, col_start:col_start + 3].flatten()))
                        for row_start in range(0, 9, 3) for col_start in range(0, 9, 3))
        return row_score + col_score + box_score

    def generate_individual(self):
        def fill_grid(grid):
            """Backtracking to fill the grid."""
            for row in range(9):
                for col in range(9):
                    if grid[row, col] == 0:
                        for num in range(1, 10):
                            if self.is_safe(grid, row, col, num):
                                grid[row, col] = num
                                if fill_grid(grid):
                                    return True
                                grid[row, col] = 0
                        return False
            return True

        individual = self.puzzle.copy()
        fill_grid(individual)
        return individual

    def is_safe(self, grid, row, col, num):
        """Check if a number can be placed at (row, col)."""
        if num in grid[row, :] or num in grid[:, col]:
            return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row + 3, box_col:box_col + 3]:
            return False
        return True

    def initialize_population(self):
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for row in range(9):
            if random.random() > 0.5:
                child[row, :] = parent2[row, :]
        return child

    def mutate(self, individual):
        for row in range(9):
            if random.random() < self.mutation_rate:
                unfixed_cols = [col for col in range(9) if not self.is_fixed(row, col)]
                if len(unfixed_cols) > 1:
                    col1, col2 = random.sample(unfixed_cols, 2)
                    individual[row, col1], individual[row, col2] = individual[row, col2], individual[row, col1]
        return individual

    def evolve(self):
        new_population = []
        sorted_population = sorted(self.population, key=lambda ind: -self.fitness(ind))
        self.best_solution = sorted_population[0]
        new_population.extend(sorted_population[:self.population_size // 10])  # Elitism

        while len(new_population) < self.population_size:
            parent1, parent2 = random.choices(sorted_population[:100], k=2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def solve(self):
        self.initialize_population()
        for generation in range(self.generations):
            self.evolve()
            best_fitness = self.fitness(self.best_solution)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            if best_fitness == 243:  # Max fitness (81 rows + 81 columns + 81 boxes)
                break
        return self.best_solution