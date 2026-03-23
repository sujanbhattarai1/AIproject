import numpy as np  

def generate_cities(num_cities, random_seed):
    """
    Randonly generate num_cities poinr in unit square [0,1] x [0,1]

    Returns:
        coordinate : (num_cities, 2) array of (x, y) positions
        distance_matrix : (num_cities, num_cities) pairwise Euclidean distance
    """

    rng = np.random.default_rng(random_seed)
    coordinates = rng.random((num_cities, 2))

    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for k in range(num_cities):
            distance_matrix[i][k] = np.linalg.norm(coordinates[i] - coordinates[k])

    return coordinates, distance_matrix
