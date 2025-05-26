import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import pandas as pd
import numpy as np


from explanations_cfl import (
    generate_instance, instance_coeff, get_preference_profiles,
    max_entropy_empirical, max_entropy_theorical, entropy_from_profiles,
    CFL_MILP_h, RelativeExplanationsMixedFacility, RelativeExplanationsDistGeneric_a,
    feasible_solution_facility, WassersteinDist, 
    get_probability_a, retrieve_feature
)



def run_experiment(D_sizes, N_sizes, r_sizes, E_size, seeds, alpha, distF, beta_d, lambs, output_file):
    results = []
    n_experiments = len(D_sizes) * len(N_sizes) * len(r_sizes) * len(seeds) * len(lambs)
    experiment_counter = 0

    for N in N_sizes:
        for D in D_sizes:
            for r in r_sizes:
                for lam in lambs:
                    for seed in seeds:
                        experiment_counter += 1
                        print(f"\n🔹 Experiment {experiment_counter}/{n_experiments}, seed {seed} "
                              f"(D={D}, N={N}, r={r}, lamW={lam})")

                        # Regularization parameters
                        lams = {"L1": 1, "W": lam}

                        # Instance
                        ins = generate_instance(D_size=D, E_size=E_size, N_size=N, seed=seed)
                        instance = instance_coeff(ins, beta_d, np.zeros(len(ins['D'])), np.zeros(len(ins['E'])))

                        # Entropy
                        profiles = get_preference_profiles(instance, beta_d=0.5,
                                                           features_d=np.zeros(len(ins['D'])),
                                                           features_e=np.zeros(len(ins['E'])),
                                                           num_samples=1000)
                        max_entropy_emp = max_entropy_empirical(profiles)
                        max_entropy_the = max_entropy_theorical(instance)
                        entropy_value = entropy_from_profiles(profiles)

                        # Solving original problem
                        print(f"\n➡️ Solving original CFL {experiment_counter}/{n_experiments}, seed {seed}")
                        z_sol, w_sol, captured_demand = CFL_MILP_h(instance, r)
                        P0 = get_probability_a(instance, z_sol)

                        # Defining Dspace
                        dspace = {k: 1 for k, v in z_sol.items() if v == 0}
                        if dspace:
                            dspace = {list(dspace.keys())[0]: 1}
                        else:
                            dspace = {}

                        # Lower bound with model-free case
                        print(f"\n➡️ Wasserstein Lower Bound {experiment_counter}/{n_experiments}, seed {seed}")
                        z_sol_gen, Prob_sol_gen, captured_new_gen, Wbound = RelativeExplanationsDistGeneric_a(
                            instance, r, z_sol, alpha, dspace, distF)

                        # Feasible solution
                        print(f"\n➡️ Calculating feasible solution {experiment_counter}/{n_experiments}, semilla {seed}")
                        initial_sol = feasible_solution_facility(instance, r, z_sol, alpha, dspace, captured_demand, distF, "I")

                        # Relative Counterfactual
                        print(f"\n➡️ Solving counterfactual problem {experiment_counter}/{n_experiments}, semilla {seed}")
                        z_sol_new, a_new, w_new, captured_demand_new, cost, runtimeMixedF, timersMixedF, GapMixedF = RelativeExplanationsMixedFacility(
                            instance, r, z_sol, alpha, dspace, Wbound, initial_sol, lams)

                        # Getting metrics
                        if z_sol_new and a_new:
                            instance2 = copy.deepcopy(instance)
                            instance2["a"] = {(n, d): a * a_new[d] for (n, d), a in instance["a"].items()}
                            P_new = get_probability_a(instance2, z_sol_new)
                            Wdist, _ = WassersteinDist(instance, P0, P_new, distF)

                            new_features = retrieve_feature(instance, a_new, beta_d, "facility")
                            new_features_r = {k: round(v, 6) for k, v in new_features.items()}
                            features_non_zero = {k: v for k, v in new_features_r.items() if v != 0}
                            sparsity_facility = len(features_non_zero) / D

                            new_profiles = get_preference_profiles(instance, beta_d=0.5,
                                                                   features_d=np.array(list(new_features.values())),
                                                                   features_e=np.zeros(len(ins['E'])),
                                                                   num_samples=1000)
                            new_entropy_value = entropy_from_profiles(new_profiles)
                        else:
                            Wdist = np.nan
                            features_non_zero = {}
                            sparsity_facility = np.nan
                            new_entropy_value = np.nan

                        # Saving results
                        results.append({
                            "Exp": experiment_counter,
                            "Initial_sol": "I",
                            "D_size": D,
                            "E_size": E_size,
                            "N_size": N,
                            "r_size": r,
                            "Seed": seed,
                            "lam": lam,
                            "OG Captured Demand": captured_demand,
                            "z_sol": z_sol,
                            "dspace": str(dspace),
                            "Wbound": Wbound,
                            "New Captured Demand": captured_demand_new,
                            "Wdistance": Wdist,
                            "Features": str(features_non_zero),
                            "Sparsity Facility": sparsity_facility,
                            "Original Entropy": entropy_value,
                            "Max entropy empirical": max_entropy_emp,
                            "Max entropy theorical": max_entropy_the,
                            "New entropy": new_entropy_value,
                            "Runtime": runtimeMixedF,
                            "Time Best Solution Mixed": timersMixedF.get("best_solution_time", None),
                            "Time Best Bound Mixed": timersMixedF.get("last_lower_bound_time", None),
                            "Gap": GapMixedF
                        })

    # Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\n✅ Results saved in: {output_file}")


if __name__ == "__main__":
    # General parameters
    # D_sizes, N_sizes, r_sizes : lists 
    D_sizes = [10,20]
    N_sizes = [100,200]
    r_sizes = [4,8]
    E_size = 5
    seeds = list(range(1, 11))
    alpha = 1
    distF = "euclidean"
    beta_d = 0.1
    lambs = [0,0.1,1]
    output_file = "experiments_26may.xlsx"

    run_experiment(D_sizes, N_sizes, r_sizes, E_size, seeds, alpha, distF, beta_d, lambs, output_file)
