import numpy as np

from src.neural_network_population_trained import NeuralNetwork

def particle_swarm_optimization(network: NeuralNetwork, X, y, 
        pop_size: int = 20, inertia: float = 0.7, cognitive_update_rate: float = 1.5, 
        social_update_rate: float = 1.5, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)

    population = initialize_population(network.layer_sizes, pop_size, rng) # Laplace Distribution
    # TODO - Initialize velocities
    
    """
    repeat
        for each particle i = 1,...,n_s do
            // set the personal best position
            if f(x_i) < f(y_i) then
                y_i = x_i
            end
            // set the global best position
            if f(y_i) < f(g) then
                g = y_i
            end
        end
        for each particle i = 1,...,n_s do
            update the velocity
            update the position
        end
    until stopping condition is true
    """
    pass

def initialize_population(layer_sizes, pop_size, rng):
    segments = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        W = rng.laplace(0.0, 1.0, size=(pop_size, n_in * n_out))
        b = rng.laplace(0.0, 1.0, size=(pop_size, n_out))
        segments += [W, b]
    return np.hstack(segments)


def evaluate_fitness(population, network, X, y):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = network.fitness(population[i], X, y)
    return fitness