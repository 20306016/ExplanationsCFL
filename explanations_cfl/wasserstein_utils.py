import numpy as np
import ot

def distFn(instance):
    """
    Compute the squared Euclidean distances between all pairs of facilities and competitive locations.

    This function calculates and returns a dictionary containing the squared distances between:
    - Each pair of candidate facility locations in D
    - Each pair of competitve locations in E
    - Each pair formed by one candidate facility and one competitive location

    Args:
        instance (dict): Dictionary containing instance data with coordinates:
            - "D" (list): List of candidate facility identifiers.
            - "E" (list): List of competitive facility identifiers.
            - "coord_D" (list or array): Coordinates corresponding to locations in D.
            - "coord_E" (list or array): Coordinates corresponding to locations in E.

    Returns:
        dict: A dictionary where keys are tuples (location1, location2) and values are squared Euclidean distances.
    """
    D = instance["D"]
    E = instance ["E"]
    coord_D = instance["coord_D"]  
    coord_E = instance["coord_E"]   

    dist_squared = {
        **{  
            (D[i], D[j]): np.sum((coord_D[i] - coord_D[j]) ** 2)
            for i in range(len(D)) for j in range(len(D))
        },
        **{  
            (E[i], E[j]): np.sum((coord_E[i] - coord_E[j]) ** 2)
            for i in range(len(E)) for j in range(len(E))
        },
        **{  
            (D[i], E[j]): np.sum((coord_D[i] - coord_E[j]) ** 2)
            for i in range(len(D)) for j in range(len(E))
        },
        **{  
            (E[j], D[i]): np.sum((coord_E[j] - coord_D[i]) ** 2)
            for i in range(len(D)) for j in range(len(E))
    }
    }
    
    return dist_squared


def WassersteinDistOpt(instance, Prob1, Prob2, distF):
    """
    Computes the Wasserstein distance between two discrete probability distributions over facilities,
    using the POT library (optimal transport) instead of solving an LP with Gurobi.

    Args:
        instance (dict): Same structure as in your original function, with "D", "E", "N", etc.
        Prob1 (dict): Prob1[n][c]
        Prob2 (dict): Prob2[n][c]
        distF (str): "euclidean" or "constant"

    Returns:
        tuple:
            - float: total Wasserstein distance
            - dict: transport plan pi[(n, c, c')]
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C = D + E

    # Construir la matriz de costes
    if distF == "euclidean":
        dist_dict = distFn(instance)
        M = np.array([[dist_dict[(c, c_)] for c_ in C] for c in C])
    elif distF == "constant":
        M = np.ones((len(C), len(C)))
    else:
        raise ValueError("distF must be either 'euclidean' or 'constant'")

    total_cost = 0.0
    pi_values = {}

    for n in N:
        a = np.array([Prob1[n][c] for c in C])
        b = np.array([Prob2[n][c] for c in C])

        # Normalizing
        if not np.isclose(a.sum(), 1.0):
            a = a / a.sum()
        if not np.isclose(b.sum(), 1.0):
            b = b / b.sum()

        # Solving optimal transport problem with POT
        pi_matrix = ot.emd(a, b, M)
        w_dist = np.sum(pi_matrix * M)
        total_cost += w_dist

        for i, c in enumerate(C):
            for j, c_ in enumerate(C):
                if pi_matrix[i, j] > 1e-12:  
                    pi_values[(n, c, c_)] = pi_matrix[i, j]

    return total_cost, pi_values
