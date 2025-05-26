import numpy as np
from collections import Counter
from .instance_generation import utility_function


def get_preference_profiles(instance, beta_d, features_d, features_e, num_samples=1000, seed=0):
    """
    Simulate customer preference profiles based on random utility models (Gumbel noise).

    Args:
        instance (dict): The facility location instance with coordinates and distances.
        beta_d (float): Distance sensitivity coefficient.
        features_d (ndarray): Feature vector for candidate facilities.
        features_e (ndarray): Feature vector for competitor facilities.
        num_samples (int): Number of Monte Carlo samples to draw.
        seed (int): Random seed for reproducibility.

    Returns:
        list of tuples: Each tuple represents a binary profile indicating which candidate
                        facilities are preferred over the best competitor by a customer.
    """
    np.random.seed(seed)

    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    theta_d = instance["theta_d"]
    theta_e = instance["theta_e"]

    profiles = []

    for _ in range(num_samples):
        # Sample Gumbel noise for each customer and each facility (candidate and competitor)
        eps_d = np.random.gumbel(size=(len(N), len(D)))
        eps_e = np.random.gumbel(size=(len(N), len(E)))

        for n in range(len(N)):
            # Compute utility with noise for each candidate and competitor
            u_d = [utility_function(theta_d, n, d, beta_d, features_d) + eps_d[n, d] for d in range(len(D))]
            u_e = [utility_function(theta_e, n, e, beta_d, features_e) + eps_e[n, e] for e in range(len(E))]

            # Determine the best utility among competitors
            best_competitor = max(u_e)

            # Identify which candidate facilities are preferred
            preferred_d_indices = [d for d, u in enumerate(u_d) if u > best_competitor]

            # Create a binary profile: 1 if preferred, 0 otherwise
            profile = tuple(1 if d in preferred_d_indices else 0 for d in range(len(D)))
            profiles.append(profile)

    return profiles

def entropy_from_profiles(profiles):
    """
    Compute the empirical entropy of a set of preference profiles.

    Args:
        profiles (list of tuples): Simulated preference profiles.

    Returns:
        float: Empirical entropy.
    """
    count = Counter(profiles)
    total = sum(count.values())
    probs = np.array([v / total for v in count.values()])
    return -np.sum(probs * np.log(probs))

def max_entropy_empirical(profiles):
    """
    Compute the maximum possible entropy given the number of unique profiles observed.

    Args:
        profiles (list of tuples): Simulated preference profiles.

    Returns:
        float: Maximum empirical entropy.
    """
    num_profiles = len(set(profiles))  # Number of unique profiles
    return np.log(num_profiles) if num_profiles > 0 else 0.0


def max_entropy_theorical(instance):
    """
    Compute the theoretical upper bound of entropy for the system.

    Args:
        instance (dict): The instance dictionary.

    Returns:
        float: Theoretical maximum entropy assuming all outcomes are equally likely.
    """
    N = instance["N"]
    D = instance["D"]
    return len(N) * len(D) * np.log(2)

