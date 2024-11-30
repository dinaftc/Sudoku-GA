import numpy as np
import random
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

Nd = 9  # Sudoku grid size

class Candidate:
    def __init__(self, initial):
        self.values = np.zeros((Nd, Nd), dtype=int)
        self.fitness = None
        self.initial = initial  # Stores the fixed values in the Sudoku grid

    def initialize(self):
        """Initialize the candidate by filling missing values with valid numbers."""
        for i in range(Nd):
            for j in range(Nd):
                if self.initial[i][j] != 0:
                    self.values[i][j] = self.initial[i][j]
                else:
                    # Valid values for this cell (no duplicates in row, column, or subgrid)
                    possible_values = list(
                        set(range(1, Nd + 1)) -
                        set(self.values[i]) -  # Row
                        set(self.values[:, j]) -  # Column
                        set(self.values[i // 3 * 3:i // 3 * 3 + 3, j // 3 * 3:j // 3 * 3 + 3].flatten())  # Subgrid
                    )
                    if possible_values:
                        self.values[i][j] = random.choice(possible_values)
                    else:
                        # Fallback if no values are valid (can occur during initialization)
                        self.values[i][j] = random.randint(1, Nd)


    def update_fitness(self):
        """Calculate fitness based on the number of duplicates."""
        conflicts = 0
        # Row and column conflicts
        for i in range(Nd):
            conflicts += (Nd - len(set(self.values[i])))  # Row conflicts
            conflicts += (Nd - len(set(self.values[:, i])))  # Column conflicts

        # Subgrid conflicts
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                subgrid = self.values[i:i+3, j:j+3].flatten()
                conflicts += (Nd - len(set(subgrid)))

        # Assign fitness (inverse of conflicts, higher is better)
        self.fitness = 1 / (1 + conflicts) if conflicts > 0 else float('inf')  # Perfect fitness for no conflicts


    def mutate(self, mutation_rate):
        """Mutate the candidate by swapping two valid numbers in a row."""
        if random.random() < mutation_rate:
            row = random.randint(0, Nd - 1)
            mutable_positions = [j for j in range(Nd) if self.initial[row][j] == 0]
            if len(mutable_positions) >= 2:
                pos1, pos2 = random.sample(mutable_positions, 2)
                self.values[row][pos1], self.values[row][pos2] = self.values[row][pos2], self.values[row][pos1]
        self.update_fitness()



class Population:
    def __init__(self, initial):
        self.candidates = []
        self.initial = initial

    def seed(self, Nc):
        """Seed the population with Nc candidates."""
        for i in range(Nc):
            candidate = Candidate(self.initial)
            candidate.initialize()
            candidate.update_fitness()
            self.candidates.append(candidate)
            logging.debug(f"Seeded candidate {i} with fitness {candidate.fitness}")

    def update_fitness(self):
        """Update fitness for all candidates."""
        for candidate in self.candidates:
            candidate.update_fitness()

    def evolve(self, crossover_rate, mutation_rate):
        """Generate a new population using crossover, mutation, and occasional reseeding."""
        new_population = self.select_elites(len(self.candidates) // 10)
        while len(new_population) < len(self.candidates):
            parent1, parent2 = random.sample(self.candidates, 2)
            child1, child2 = self.crossover(parent1, parent2, crossover_rate)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            child1.update_fitness()
            child2.update_fitness()
            new_population.extend([child1, child2])
        
        # Occasionally introduce new random candidates to avoid stagnation
        while len(new_population) < len(self.candidates):
            candidate = Candidate(self.initial)
            candidate.initialize()
            candidate.update_fitness()
            new_population.append(candidate)
        
        self.candidates = new_population[:len(self.candidates)]


    def select_elites(self, Ne):
        """Select the top Ne candidates."""
        self.candidates.sort(key=lambda c: c.fitness, reverse=True)
        return self.candidates[:Ne]

    def crossover(self, parent1, parent2, crossover_rate):
        """Perform crossover by mixing subgrids from two parents."""
        child1 = Candidate(self.initial)
        child2 = Candidate(self.initial)
        if random.random() < crossover_rate:
            for i in range(0, Nd, 3):  # Subgrid row blocks
                if random.random() < 0.5:
                    child1.values[i:i+3] = parent1.values[i:i+3]
                    child2.values[i:i+3] = parent2.values[i:i+3]
                else:
                    child1.values[i:i+3] = parent2.values[i:i+3]
                    child2.values[i:i+3] = parent1.values[i:i+3]
        else:
            child1.values = parent1.values.copy()
            child2.values = parent2.values.copy()
        return child1, child2



def genetic_algorithm(initial, Nc=1000, Ng=1000, mutation_rate=0.1, crossover_rate=0.8):
    population = Population(initial)
    population.seed(Nc)
    best_fitness = 0
    stagnation_count = 0

    for generation in range(Ng):
        population.update_fitness()
        best_candidate = max(population.candidates, key=lambda c: c.fitness)


        if best_candidate.fitness > best_fitness:
            best_fitness = best_candidate.fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        if 1 / (1 + (1 / best_candidate.fitness - 1)) == 0:  # No conflicts, solution found
            logging.info(f"Solution found at generation {generation}")
            return best_candidate.values
       
        if best_candidate.fitness == float('inf'):  # Perfect solution found
            logging.info(f"Solution found at generation {generation}")
            return best_candidate.values
        
        if stagnation_count > 100:  # Reseed if stagnation occurs
            logging.info(f"Stagnation detected at generation {generation}. Reseeding population.")
            population.seed(Nc)
            stagnation_count = 0
        
        population.evolve(crossover_rate, mutation_rate)

        if generation % 100 == 0:
            logging.info(f"Generation {generation}: Best fitness = {best_fitness}")

    return max(population.candidates, key=lambda c: c.fitness).values
