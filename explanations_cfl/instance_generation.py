import matplotlib.pyplot as plt
import numpy as np
import copy

def generate_instance(D_size, E_size, N_size, width=20, seed=0):
    """
    Generate a random instance of the facility location problem.

    Args:
        D_size (int): Number of candidate facility locations.
        E_size (int): Number of existing competitive facilities.
        N_size (int): Number of customers.
        width (int): Size of the square area for spatial distribution.
        seed (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing coordinates, distances, weights, and sets.
    """
    np.random.seed(seed)

    # Generate random coordinates for facilities and customers
    coord_E = width * np.random.rand(E_size, 2)  # Competitive facilities
    coord_D = width * np.random.rand(D_size, 2)  # Candidate locations
    coord_N = width * np.random.rand(N_size, 2)  # Customers

    # Compute L1 (Manhattan) distances from customers to facilities
    theta_d = np.abs(coord_N[:, np.newaxis] - coord_D).sum(axis=2)
    theta_e = np.abs(coord_N[:, np.newaxis] - coord_E).sum(axis=2)

    theta_d = np.round(theta_d, decimals=3)
    theta_e = np.round(theta_e, decimals=3)

    # All customers have unit demand
    q = np.ones(N_size, dtype=int)

    # Generate identifiers for each element
    D = [f"d{i+1}" for i in range(D_size)]
    E = [f"e{i+1}" for i in range(E_size)]
    N = [f"n{i+1}" for i in range(N_size)]

    return {
        "D": D,
        "E": E,
        "N": N,
        "coord_D": coord_D,
        "coord_E": coord_E,
        "coord_N": coord_N,
        "theta_d": theta_d,
        "theta_e": theta_e,
        "q": q
    }


def visualize_instance(instance, facility_status=None):
    """
    Visualize an instance of the facility location problem.

    Args:
        instance (dict): Dictionary containing coordinates and identifiers.
        facility_status (dict, optional): Binary status indicating whether each candidate location is selected.
                                          Expected keys are 'd1', 'd2', ..., with values 1.0 or 0.0.
    """
    D = len(instance["D"])
    N = len(instance["N"])
    E = len(instance["E"])

    plt.figure(figsize=(6, 6))
    
    # Extract coordinates
    pos_D = {facility: pos for facility, pos in zip(instance['D'], instance['coord_D'])}
    pos_E = {facility: pos for facility, pos in zip(instance['E'], instance['coord_E'])}
    pos_N = {customer: pos for customer, pos in zip(instance['N'], instance['coord_N'])}

    # Plot candidate locations
    for node, coords in pos_D.items():
        facility_id = node  # e.g., 'd1'
        color = 'black' if facility_status and facility_status.get(facility_id) == 1.0 else 'white'
        plt.scatter(*coords, color=color, edgecolor='black', s=200, marker='^')
    
    # Plot competitive facilities
    for node, coords in pos_E.items():
        plt.scatter(*coords, color='black', s=200, marker='D')
    
    # Plot customers
    for node, coords in pos_N.items():
        plt.scatter(*coords, color='black', s=80, marker='o')  

    # Annotate nodes
    for node, coords in {**pos_D, **pos_E, **pos_N}.items():
        plt.text(coords[0], coords[1] + 0.5, node, fontsize=10, ha='center', color='black')
    
    plt.title("Instance Visualization")
    plt.axis('equal') 
    plt.savefig(f"instance_D_{D}E_{E}N_{N}.png", dpi=300, bbox_inches='tight')
    plt.show()


def utility_function(theta, n, c, beta_d, features):
    """
    Compute the utility of assigning customer n to facility/location c.

    Args:
        theta (ndarray): Distance matrix (theta_d or theta_e).
        n (int): Index of the customer.
        c (int): Index of the facility/location.
        beta_d (float): Distance sensitivity coefficient.
        features (ndarray): Vector of facility-level attributes (e.g., price, quality).

    Returns:
        float: Utility score for assigning customer n to facility c.
    """
    return -beta_d * theta[n, c] + features[c]


def instance_coeff(instance, beta_d, features_d, features_e):
    """
    Augment the instance with coefficients derived from utility functions.

    Args:
        instance (dict): The base instance.
        beta_d (float): Distance coefficient.
        features_d (ndarray): Attributes of candidate facilities.
        features_e (ndarray): Attributes of competitive facilities.

    Returns:
        dict: An extended copy of the instance including:
              - 'a': customer-to-candidate utility exponentials,
              - 'b': customer-to-competitor utility sums,
              - 'be': customer-to-competitor utility exponentials (by pair).
    """
    instance2 = copy.deepcopy(instance)

    instance2["a"] = {
        (instance["N"][n], instance["D"][d]): np.exp(utility_function(instance["theta_d"], n, d, beta_d, features_d))
        for n in range(len(instance["N"]))
        for d in range(len(instance["D"]))
    }

    instance2["b"] = {
        instance["N"][n]: sum(
            np.exp(utility_function(instance["theta_e"], n, e, beta_d, features_e))
            for e in range(len(instance["E"]))
        )
        for n in range(len(instance["N"]))
    }

    instance2["be"] = {
        (instance["N"][n], instance["E"][e]): np.exp(utility_function(instance["theta_e"], n, e, beta_d, features_e))
        for n in range(len(instance["N"]))
        for e in range(len(instance["E"]))
    }

    return instance2

