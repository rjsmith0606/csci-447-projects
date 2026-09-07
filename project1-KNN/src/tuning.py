"""
Hyperparameter tuning for the KNN algorithm.
"""
from evaluation import cross_validation_loop  # Use your existing CV logic


def tune_knn_params(features, labels, task='classification'):
    # 1. Define the grid of values to test
    k_candidates = [1, 3, 5, 7, 11]
    gamma_candidates = [0.1, 1.0, 10.0] if task == 'regression' else [None]

    best_score = float('inf')
    best_params = {}

    # 2. Nested loops to test every combination
    for k in k_candidates:
        for gamma in gamma_candidates:
            # Call your 5x2 cross-validation loop
            # Ensure your CV loop can accept k and gamma as arguments
            current_score = cross_validation_loop(features, labels, model=task, k=k, gamma=gamma)

            # 3. Track the best performing combination
            if current_score < best_score:
                best_score = current_score
                best_params = {'k': k, 'gamma': gamma}

    return best_params
