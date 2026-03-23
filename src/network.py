import numpy as np 

def sigmoid(x, gain=0.5):
    """sigmoid function"""
    return 1 / (1 + np.exp(-gain * x))

def compute_energy(activation, distance_matrix, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    """
    The Hopfield energy function 
    Sum of four penalty term
    A valid tour minimises all four simultaneously

    Args:
        activation: (num_cities, num_cities) matrix V[i][j] 
                    V[i][j] ~ 1 mean city i is visited at position j 
        deistance_matrix: pairwise city distance

    Returns:
        scalar energy value 
    """
    N = num_cities
    V = activation

    # each city visited exactly once
    # penalise rows whose sum differ from 1
    row_penalty = penalty_row / 2 * np.sum((np.sum(V, axis=1) - 1) ** 2)

    # one city per time slot
    # penalises column whose sum differs from 1
    col_penalty = penalty_col / 2 * np.sum((np.sum(V, axis=0) - 1) ** 2)

    ## add edge weight d[i][k] when cirty i is at position j and city k is at the next or previous position
    distance_penalty = penalty_distance / 2 * np.sum([
        distance_matrix[i, k] * V[i, j] * (V[k, (j + 1) % N] + V[k, (j - 1) % N]) for i in range(N) for j in range(N) for k in range(N)
    ])

    # total activation should be exactly equal to N 
    toursize_penalty = penalty_toursize / 2 * (np.sum(V) - N) ** 2

    return row_penalty + col_penalty + distance_penalty + toursize_penalty


def compute_net_input(activation, distance_matrix, i, j, num_cities, penalty_row, penalty_col, penalty_distance, penalty_toursize):
    """
    Net input to neuron (i, j) is negative gradient w.r.t V[i][j]
    Positive net input => neuron activate
    Negative net input => neuron switch off
    """
    N = num_cities
    V = activation

    # discourage activating if row i is already full
    from_row = -penalty_row * (np.sum(V[i, :]) - V[i, j] - 1)

    # discouragge activating if column j is already full
    from_col = -penalty_col * (np.sum(V[:, j]) - V[i, j] - 1)

    # reward connecting to nearby cities at adjacent positions
    from_distance = -penalty_distance * np.sum ([
        distance_matrix[i, k] * (V[k, (j + 1) % N] + V[k, (j - 1) % N]) for k in range(N)
    ])

    # discourage activating if too many neuron are already on
    from_toursize = -penalty_toursize * (np.sum(V) - V[i, j] - N)

    return from_row + from_col + from_distance + from_toursize 


def run_hopfield(num_cities, distance_matrix, random_seed, penalty_row, penalty_col, penalty_distance, penalty_toursize, step_size, time_const, max_iter):
    """
    Run asynchronous Hopfield update till max_iter

    At each step:
        1. Pick random neuron (city i, position j)
        2. Compute net input from other neurons
        3. Update its membrane potential u[i][j]
        4. squash u through sigmoid to get new_activation

    Returns:
        activation : final (num_cities, num_cites) V matrix
        energy_history : list of energy values sampled every 100 steps
    """
    N = num_cities
    rng = np.random.default_rng(random_seed)

    # initialize membrane potential neat 1/N so all neuron start weakly active
    membrane_potential = rng.uniform(-0.1, 0.1, (N, N)) + (1/N)
    activation = sigmoid(membrane_potential)

    energy_history = []

    for step in range(max_iter):
        # one random neuron update per step
        i = rng.integers(0, N)
        j = rng.integers(0, N)

        net = compute_net_input(activation, distance_matrix, i, j, N, penalty_row, penalty_col, penalty_distance, penalty_toursize)

        # Differential equation : u decays toward 0 and is driven by nwt input
        membrane_potential[i, j] += step_size * (-membrane_potential[i, j] / time_const + net)
        activation[i, j] = sigmoid(membrane_potential[i, j])

        if step % 100 == 0:
            energy = compute_energy(activation, distance_matrix, N, penalty_row, penalty_col, penalty_distance, penalty_toursize)
            energy_history.append(energy)

    return activation, energy_history
