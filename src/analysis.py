import numpy as np  
from src.network import run_hopfield

def is_valid_tour(activation, num_cities, threshold=0.5):
    """
    Binarise the activation matrix and check it it encodes a valid tour
    A valid tour has exactly one 1 per row and exactly one 1 per column

    Returns:
        is_valid : bool
        binary_V : binarised activation matrix
    """
    binary_activation = (activation > threshold).astype(int)

    each_city_visited_once = all(
        np.sum(binary_activation[i, :]) == 1 for i in range(num_cities)
    )

    one_city_per_slot = all(
        np.sum(binary_activation[:, j]) == 1 for j in range(num_cities)
    )

    return each_city_visited_once and one_city_per_slot, binary_activation


def decode_tour(binary_activation, num_cities):
    """
    Extract city visited order from a valid binary activation matrix
    Read which city is active at each position j, left to right

    Returns:
        tour: list of city indices, length num_cities + 1 
    """
    tour = []
    for position in range(num_cities):
        city = np.argmax(binary_activation[:, position])
        tour.append(city)
    tour.append(tour[0])
    return tour

def compute_tour_distance(tour, distance_matrix):
    """sum up egde distance along the decoded tour"""
    return sum(
        distance_matrix[tour[step], tour[step + 1]]
        for step in range(len(tour) - 1)
    )

def measure_success_rate(num_trials, num_cities, distance_matrix, penalty_row, penalty_col, penalty_distance, penalty_toursize, step_size, time_const, max_iter):
    """
    Run Hopfield network num_trial times with different random seeds and
    count how often a valid tour is found
    """
    successes = 0
    for trial in range(num_trials):
        activation, _ = run_hopfield(num_cities, distance_matrix, random_seed=trial, penalty_row=penalty_row, penalty_col=penalty_col, penalty_distance=penalty_distance, penalty_toursize=penalty_toursize, step_size=step_size, time_const=time_const, max_iter=max_iter)

        valid, _ = is_valid_tour(activation, num_cities)
        if valid:
            successes += 1

    print(f"Valid tours: {successes}/{num_trials}  ({100 * successes / num_trials:.0f}%)")
    return successes / num_trials

    