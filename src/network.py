# import numpy as np
#
# def sigmoid(x, gain=1.0):
#     return 1 / (1 + np.exp(-gain * x))
#
# def compute_energy(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
#     N = num_cities
#     V = activation
#
#     row_penalty      = penalty_row      / 2 * np.sum((np.sum(V, axis=1) - 1) ** 2)
#     col_penalty      = penalty_col      / 2 * np.sum((np.sum(V, axis=0) - 1) ** 2)
#     toursize_penalty = penalty_toursize / 2 * (np.sum(V) - N) ** 2
#
#     V_next = np.roll(V, -1, axis=1)
#     V_prev = np.roll(V,  1, axis=1)
#     distance_penalty = penalty_distance / 2 * np.einsum('ij,ik,kj->', V, distance_matrix, V_next + V_prev)
#
#     return row_penalty + col_penalty + distance_penalty + toursize_penalty
#
# def compute_all_net_inputs(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
#     N = num_cities
#     V = activation
#
#     row_sums = np.sum(V, axis=1, keepdims=True)
#     col_sums = np.sum(V, axis=0, keepdims=True)
#     total    = np.sum(V)
#     from_row      = -penalty_row      * (row_sums - V - 1)
#     from_col      = -penalty_col      * (col_sums - V - 1)
#     from_toursize = -penalty_toursize * (total - V - N)
#
#     V_next = np.roll(V, -1, axis=1)
#     V_prev = np.roll(V,  1, axis=1)
#     from_distance = -penalty_distance * (distance_matrix @ (V_next + V_prev))
#
#     return from_row + from_col + from_distance + from_toursize
#
# def run_hopfield(num_cities, distance_matrix, random_seed, penalty_row, penalty_col, penalty_distance, penalty_toursize, step_size, time_const, max_iter):
#     N   = num_cities
#     rng = np.random.default_rng(random_seed)
#
#     membrane_potential = rng.uniform(-0.1, 0.1, (N, N)) + (1 / N)
#     activation         = sigmoid(membrane_potential)
#
#     energy_history = []
#
#     gain_start = 0.5
#     gain_end   = 10.0
#
#     for step in range(max_iter):
#         progress = step / max_iter
#         gain     = gain_start + (gain_end - gain_start) * progress
#
#         net = compute_all_net_inputs(
#             activation, distance_matrix, N,
#             penalty_row, penalty_col, penalty_distance, penalty_toursize
#         )
#
#         membrane_potential += step_size * (-membrane_potential / time_const + net)
#         activation          = sigmoid(membrane_potential, gain=gain)
#
#         if step % 100 == 0:
#             energy = compute_energy(
#                 activation, distance_matrix, N,
#                 penalty_row, penalty_col, penalty_distance, penalty_toursize
#             )
#             energy_history.append(float(energy))
#
#     return activation, energy_history

import numpy as np

def sigmoid(x, gain=1.0):
    return 1 / (1 + np.exp(-gain * x))

def compute_energy(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    N = num_cities
    V = activation

    row_penalty      = penalty_row      / 2 * np.sum((np.sum(V, axis=1) - 1) ** 2)
    col_penalty      = penalty_col      / 2 * np.sum((np.sum(V, axis=0) - 1) ** 2)
    toursize_penalty = penalty_toursize / 2 * (np.sum(V) - N) ** 2

    V_next = np.roll(V, -1, axis=1)
    V_prev = np.roll(V,  1, axis=1)
    distance_penalty = penalty_distance / 2 * np.einsum('ij,ik,kj->', V, distance_matrix, V_next + V_prev)

    return row_penalty + col_penalty + distance_penalty + toursize_penalty

def compute_all_net_inputs(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    N = num_cities
    V = activation

    row_sums = np.sum(V, axis=1, keepdims=True)
    col_sums = np.sum(V, axis=0, keepdims=True)
    total    = np.sum(V)
    from_row      = -penalty_row      * (row_sums - V - 1)
    from_col      = -penalty_col      * (col_sums - V  - 1)
    from_toursize = -penalty_toursize * (total - V - N)

    V_next = np.roll(V, -1, axis=1)
    V_prev = np.roll(V,  1, axis=1)
    from_distance = -penalty_distance * (distance_matrix @ (V_next + V_prev))

    return from_row + from_col + from_distance + from_toursize

def run_hopfield(num_cities, distance_matrix, random_seed, penalty_row, penalty_col, penalty_distance, penalty_toursize, step_size, time_const, max_iter):
    N   = num_cities
    rng = np.random.default_rng(random_seed)

    membrane_potential = rng.uniform(-0.1, 0.1, (N, N)) + (1 / N)
    activation         = sigmoid(membrane_potential)

    energy_history = []

    gain_start = 0.5
    gain_end   = 20.0

    for step in range(max_iter):
        progress = step / max_iter
        # exponential schedule: sharpens neurons earlier in the run
        gain     = gain_start * (gain_end / gain_start) ** progress

        net = compute_all_net_inputs(
            activation, distance_matrix, N,
            penalty_row, penalty_col, penalty_distance, penalty_toursize
        )

        membrane_potential += step_size * (-membrane_potential / time_const + net)
        activation          = sigmoid(membrane_potential, gain=gain)

        if step % 100 == 0:
            energy = compute_energy(
                activation, distance_matrix, N,
                penalty_row, penalty_col, penalty_distance, penalty_toursize
            )
            energy_history.append(float(energy))

    return activation, energy_history
