import random as rndm
import time

def make_gene(initial=None):
    if initial is None:
        initial = [0] * 9
    mapp = {}
    gene = list(range(1, 10))
    rndm.shuffle(gene)
    for i in range(9):
        mapp[gene[i]] = i
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            temp = gene[i], gene[mapp[initial[i]]]
            gene[mapp[initial[i]]], gene[i] = temp
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]
    return gene

def make_chromosome(initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    chromosome = []
    for i in range(9):
        chromosome.append(make_gene(initial[i]))
    return chromosome

def make_population(count, initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    population = []
    for _ in range(count):
        population.append(make_chromosome(initial))
    return population

def get_fitness(chromosome):
    """Calculate the fitness of a chromosome (Sudoku puzzle)."""
    fitness = 0

    # Check rows
    for row in chromosome:
        unique_numbers = set(row)
        fitness += len(unique_numbers)  # Reward for unique numbers

    # Check columns
    for col in range(9):
        seen = set()
        for row in range(9):
            seen.add(chromosome[row][col])
        fitness += len(seen)  # Reward for unique numbers in column

    # Check 3x3 sub-grids
    for grid_row in range(0, 9, 3):
        for grid_col in range(0, 9, 3):
            seen = set()
            for row in range(grid_row, grid_row + 3):
                for col in range(grid_col, grid_col + 3):
                    seen.add(chromosome[row][col])
            fitness += len(seen)  # Reward for unique numbers in grid

    # Max fitness is 243 (81 per rows + columns + grids)
    # Scale to return penalties for missing constraints
    return fitness - 243


ch = make_chromosome()
print(get_fitness(ch))


def pch(ch):
    for i in range(9):
        for j in range(9):
            print(ch[i][j], end=" ")
        print("")

def crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = rndm.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        elif x == 0:
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2

def mutation(ch, pm, initial):
    for i in range(9):
        x = rndm.randint(0, 100)
        if x < pm * 100:
            ch[i] = make_gene(initial[i])
    return ch


def r_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool

def w_get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    weight = [fit[0] - fitness_list[0][0] for fit in fitness_list]
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weights=weight)[0]
        pool.append(ch[1])
    return pool

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rndm.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool

# Population size
POPULATION = 1000

# Number of generations
REPETITION = 1000

# Probability of mutation
PM = 0.1

# Probability of crossover
PC = 0.95

def read_puzzle(puzzle):
    """
    Validates and processes a Sudoku puzzle provided as a list.

    Args:
        puzzle (list): A 9x9 list representing the Sudoku puzzle.

    Returns:
        list[list[int]]: A validated 9x9 grid representing the puzzle.

    Raises:
        ValueError: If the input list is not a valid 9x9 Sudoku puzzle.
    """
    if not isinstance(puzzle, list):
        raise ValueError("The puzzle must be a list.")

    if len(puzzle) != 9:
        raise ValueError("The puzzle must have exactly 9 rows.")

    for row in puzzle:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("Each row in the puzzle must have exactly 9 elements.")
        if any(not isinstance(cell, int) or cell < 0 or cell > 9 for cell in row):
            raise ValueError("Each cell in the puzzle must be an integer between 0 and 9.")

    return puzzle


# Main genetic algorithm function
def genetic_algorithm(initial_file):
    initial = read_puzzle(initial_file)
    population = make_population(POPULATION, initial)
    for _ in range(REPETITION):
        mating_pool = r_get_mating_pool(population)
        rndm.shuffle(mating_pool)
        population = get_offsprings(mating_pool, initial, PM, PC)
        fit = [get_fitness(c) for c in population]
        m = max(fit)
        if m == 0:
            return population
    return population


def solve(puzzle):
    r = genetic_algorithm(puzzle)

    fit = [get_fitness(c) for c in r]
    m = max(fit)
    print('max fitness :', max(fit))

    for c in r:
        if get_fitness(c) == m:
            return c  # Return the solution as a list
    return None


