import numpy as np

from src.neural_network_population_trained import NeuralNetwork

# DE/rand/1/binomial
def differential_evolution(network: NeuralNetwork, X, y, 
        pop_size: int = 20, scaling_factor: float = 0.5, crossover_prob: float = 0.8, 
        n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)

    population = initialize_population(network.layer_sizes, pop_size, rng) # Gaussian Distribution - LeCun Initialization
    fitness = evaluate_fitness(population, network, X, y)
    
    for _ in range(n):
        for i in range(population.shape[0]):
            donor = mutation(population, i, scaling_factor, rng) # rand/1
            offspring = crossover(population[i], donor, crossover_prob, rng) # binomial

            offspring_fitness = network.fitness(offspring, X, y)
            if offspring_fitness <= fitness[i]:
                population[i] = offspring
                fitness[i] = offspring_fitness

    network.set_weights(population[np.argmin(fitness)])

    return population


def initialize_population(layer_sizes, pop_size, rng):
    segments = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        sd = 1.0 / np.sqrt(n_in)  # Standard deviation for Gaussian initialization
        W = rng.normal(0.0, sd, size=(pop_size, n_in * n_out))
        b = rng.normal(0.0, sd, size=(pop_size, n_out))
        segments += [W, b]
    return np.hstack(segments)


def evaluate_fitness(population, network, X, y):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = network.fitness(population[i], X, y)
    return fitness


def mutation(population, i, beta, rng):
    r1, r2, r3 = rng.choice([j for j in range(population.shape[0]) if j != i], size=3, replace=False)
    donor = population[r1] + beta * (population[r2] - population[r3])
    return donor


def crossover(target, donor, alpha, rng):
    mask = rng.random(target.shape[0]) < alpha
    mask[rng.integers(target.shape[0])] = True   # j_rand: guarantee at least one donor gene
    offspring = np.where(mask, donor, target)
    return offspring