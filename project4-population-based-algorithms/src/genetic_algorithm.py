import numpy as np

from src.neural_network_population_trained import NeuralNetwork

def genetic_algorithm(network: NeuralNetwork, X, y, 
        pop_size: int = 20, crossover_rate: float = 0.8, mutation_rate: float = 0.1, 
        n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    
    population = initialize_population(network.layer_sizes, pop_size, rng) # Laplace Distribution
    pop_fitness = evaluate_fitness(population, network, X, y)
    for _ in range(n):
        parents = selection(population, pop_fitness, rng) # Tournament Selection
        offspring = crossover(parents, crossover_rate, rng) # Single Point Crossover
        offspring = mutation(offspring, mutation_rate, rng) # Gaussian Mutation
        offspring_fitness = evaluate_fitness(offspring, network, X, y) 
        population, pop_fitness = replacement(population, pop_fitness, offspring, offspring_fitness, rng) # Steady-State Replacement
    return population, pop_fitness
        


def initialize_population(layer_sizes, pop_size, rng):
    segments = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        W = rng.laplace(0.0, 1.0, size=(pop_size, n_in * n_out))
        b = rng.laplace(0.0, 1.0, size=(pop_size, n_out))
        segments += [W, b]
    return np.hstack(segments)   # shape (pop_size, n_weights), order matches fitness()


def selection(population, pop_fitness, rng, k: int = 3):
    # tournament selection - K IS A HYPERPARAMETER  >:(
    parents = []
    for _ in range(2):
        candidates = rng.choice(population.shape[0], size=k, replace=False)
        parents.append(population[candidates[np.argmin(pop_fitness[candidates])]])
    return parents
    
def crossover(parents, crossover_rate, rng):
    p1, p2 = parents
    if rng.random() < crossover_rate:
        # Single-point crossover
        point = rng.integers(1, p1.shape[0])  # Choose a crossover point
        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])
    else:
        child1 = p1.copy()
        child2 = p2.copy()
    return np.array([child1, child2])


def mutation(offspring, mutation_rate, rng):
    for child in offspring:
        mask = rng.random(child.shape[0]) < mutation_rate
        child[mask] += rng.normal(0.0, 1.0, size=mask.sum())
    return offspring



def replacement(population, pop_fitness, offspring, offspring_fitness, rng, k: int = 3):
    # steady-state, inverse tournament: each child replaces the worst of k random members
    for child, f in zip(offspring, offspring_fitness):
        candidates = rng.choice(population.shape[0], size=k, replace=False)
        victim = candidates[np.argmax(pop_fitness[candidates])]
        population[victim] = child
        pop_fitness[victim] = f
    return population, pop_fitness



def evaluate_fitness(population, network, X, y):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = network.fitness(population[i], X, y)
    return fitness