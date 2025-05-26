
import numpy as np

def get_probability_a(instance, z_sol):
    """
    Compute the allocation probabilities for each demand point over open facilities and competitive options.

    For each demand point `n`, this function computes the probability that demand is allocated to each
    open facility `d` in `D` and each competitive facility `e` in `E`, based on a utility-like function
    involving the coefficients `a`, `b`, and `be`. Probabilities are normalized over the total utility
    available to each `n`.

    Args:
        instance (dict): Dictionary containing instance data, including:
            - "N" (list): List of demand points.
            - "D" (list): List of candidate facility locations.
            - "E" (list): List of competitive facility locations.
            - "a" (dict): Coefficients (n, d) representing utility of assigning demand `n` to facility `d`.
            - "b" (dict): Coefficients (n) representing sum of utility of assigning demand `n` to all competitive facilities.
            - "be" (dict): Coefficients (n, e) representing utility of assigning demand `n` to competitive facility `e`.
            - "q" (list): Demand quantity for each `n`.
        z_sol (dict): Dictionary indicating which facilities in `D` are open (binary values).

    Returns:
        dict: A nested dictionary `Prob` such that:
            - `Prob[n][d]` is the probability that demand point `n` is assigned to facility `d` (in `D`)
            - `Prob[n][e]` is the probability that demand point `n` is assigned to competitive facility `e` (in `E`)

    Notes:
        - Probabilities are computed using a softmax-style normalization based on utility coefficients.
        - If `z_sol[d] == 0`, the corresponding facility contributes 0 to the utility denominator.
        - The sum of probabilities over `D` and `E` for each `n` equals 1.
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    a=instance["a"]
    b=instance["b"]
    be=instance["be"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    
    Prob = { N[i]: {
        **{D[d]: (a[N[i], D[d]] * z_sol[D[d]]) / (sum(a[N[i], D[k]] * z_sol[D[k]] for k in range(len(D))) + b[N[i]])
            for d in range(len(D))},
        **{E[e]: (be[N[i],E[e]] / (sum(a[N[i], D[k]] * z_sol[D[k]] for k in range(len(D))) + b[N[i]]))
            for e in range(len(E)) } 
            } for i in range(len(N))}
    
    return Prob


def retrieve_feature(instance, a_sol, beta_d, type):
    """
    Compute features based on the solution 'a_sol' and instance parameters.

    Args:
        instance (dict): Problem instance containing sets and parameters (e.g. N, D, theta_d).
        a_sol (dict): Solution values for 'a' parameters, indexed by (n,d) or d depending on 'type'.
        beta_d (float): Parameter multiplier used when computing customer features.
        type (str): Specifies the feature type, either 'customer' or 'facility'.

    Returns:
        dict: Features computed as:
            - For 'customer': feature[(n,d)] = log(a_sol[(n,d)]) + beta_d * theta_d[i,j]
            - For 'facility': feature[d] = log(a_sol[d])
    """

    if type == "customer":
        # Compute features for each customer-facility pair (n,d)
        feature = {
            (n, d): np.log(a_sol[(n, d)]) + beta_d * instance["theta_d"][i, j]
            for i, n in enumerate(instance["N"])
            for j, d in enumerate(instance["D"])
        }
    elif type == "facility":
        # Compute features for each facility d
        feature = {d: np.log(a_sol[d]) for d in instance["D"]}
    else:
        raise ValueError("Invalid type argument: must be 'customer' or 'facility'")

    return feature

