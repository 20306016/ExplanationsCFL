import gurobipy as gp
from gurobipy import GRB
import numpy as np


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


def WassersteinDist(instance,Prob1,Prob2,distF):
    """
    Computes the Wasserstein distance between two discrete probability distributions over facilities.

    Formulates and solves a linear program to find the optimal transport plan `pi` minimizing the cost
    of moving distribution Prob1 to Prob2, where the cost is given by distances between locations.

    Args:
        instance (dict): Dictionary containing the instance data, including:
            - "D" (list): Candidate facilities.
            - "E" (list): Competitive facilities.
            - "N" (list): Demand points.
        Prob1 (dict): First discrete probability distribution over locations for each demand point.
                      Format: Prob1[n][c] for demand point n and location c in D ∪ E.
        Prob2 (dict): Second discrete probability distribution over locations, same format as Prob1.
        distF (str): Distance function type to use in the objective. Options:
                     - "euclidean": Use squared Euclidean distances between locations.
                     - "constant": Use constant cost 1 for any transport.

    Returns:
        tuple:
            - float: The optimal value of the Wasserstein distance.
            - dict: Optimal transport plan variables `pi`, indexed by (n, c, c_prime).

    Notes:
        - The transport plan `pi[n, c, c_prime]` satisfies marginal constraints matching Prob1 and Prob2.
        - Solves the problem using Gurobi MILP solver.
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C= D+ E


   
    model = gp.Model("Wasserstein Minimization")

    # Variables
    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")

    #Objective
    if distF=="euclidean":
        model.setObjective(gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C), GRB.MINIMIZE)
    if distF=="constant":
        model.setObjective(gp.quicksum(pi[n, c, c_prime] for n in N for c in C for c_prime in C), GRB.MINIMIZE)

    
    # Restricciones
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == Prob1[n][c]
                     for n in N for c in C), name="probability_constraint_1")
    
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c in C) == Prob2[n][c_prime]
                      for n in N for c_prime in C), name="probability_constraint_2")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal}")
        pi_values = {(n, c, c_prime): pi[n, c, c_prime].X
                     for n in N for c in C for c_prime in C}
        
    else:
        print("No optimal solution found.")

    return model.objVal, pi_values
