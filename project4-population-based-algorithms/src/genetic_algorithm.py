import numpy as np

from src.neural_network_population_trained import NeuralNetwork

def genetic_algorithm(network: NeuralNetwork, X, y, 
        pop_size: int = 10, crossover_rate: float = 0.8, mutation_rate: float = 0.1, 
        seed: int = 0):
    rng = np.random.default_rng(seed)
    t = 0
    
    population = initialize_population(network.layer_sizes, pop_size, rng)
    pop_fitness = evaluate_fitness(population, network, X, y)
    while t < 100:
        t += 1
        parents = selection(population, pop_fitness, rng)
        offspring = crossover(parents, crossover_rate, rng)
        offspring = mutation(offspring, mutation_rate, rng) # TODO - See below
        pop_fitness = evaluate_fitness(offspring, network, X, y) # TODO - See below
        population = offspring # CHANGE WITH REPLACEMENT STRATEGY
    return population, pop_fitness
        


def initialize_population(layer_sizes, pop_size, rng):
    segments = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        r = np.sqrt(6.0 / (n_in + n_out))                       # Xavier uniform bound
        W = rng.uniform(-r, r, size=(pop_size, n_in * n_out))
        b = np.zeros((pop_size, n_out))                          # or small uniform
        segments += [W, b]
    return np.hstack(segments)   # shape (pop_size, n_weights), order matches fitness()


def selection(population, pop_fitness, rng, k: int = 3):
    # tournament selection - K IS A HYPERPARAMETER  >:(
    selected_chromosomes = np.empty_like(population)
    for i in range(population.shape[0]):
        candidates = rng.choice(population.shape[0], size=k, replace=False)
        selected_chromosomes[i] = population[candidates[np.argmin(pop_fitness[candidates])]]
    return selected_chromosomes
    
def crossover(population, crossover_rate, rng):
    pass


def mutation(population, mutation_rate, rng):
    pass


def evaluate_fitness(population, network, X, y):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = network.fitness(population[i], X, y)
    return fitness