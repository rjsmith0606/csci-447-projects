import numpy as np

from src.neural_network_population_trained import NeuralNetwork

def differential_evolution(network: NeuralNetwork, X, y, 
        pop_size: int = 20, scaling_factor: float = 0.1, crossover_rate: float = 0.8, 
        n: int = 1000, seed: int = 0):
    pass