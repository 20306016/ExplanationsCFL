import sys
import os
import copy
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explanations_cfl import (
    generate_instance, instance_coeff, get_preference_profiles,
    max_entropy_empirical, max_entropy_theorical, entropy_from_profiles,
    CFL_MILP_h, RelativeExplanationsMixedFacility, RelativeExplanationsDistGeneric_a,
    feasible_solution_facility, WassersteinDistOpt,
    get_probability_a, retrieve_feature
)

def run_experiment(D_sizes, N_sizes, r_sizes, E_size, seeds, alpha, distF, beta_d, lambs, output_file):
    # Cargar resultados previos si existen
    if os.path.exists(output_file):
        df_prev = pd.read_excel(output_file)
        
        done = set(
            tuple(x) for x in df_prev[["D_size", "N_size", "r_size", "Seed", "lam"]].values
        )
        print(f"🔁 {len(done)} combinations already done in {output_file}.")
    else:
        df_prev = pd.DataFrame()
        done = set()
        print(f"🆕 No file {output_file} found, starting from zero.")

    # Crear lista total de combinaciones
    all_combinations = [
        (D, N, r, seed, lam)
        for D in D_sizes
        for N in N_sizes
        for r in r_sizes
        for seed in seeds
        for lam in lambs
    ]
    print(f"🧪 Total experiments: {len(all_combinations)}")
    print(f"🚧 Experiments to do: {len(all_combinations) - len(done)}")

    experiment_counter = 0

    for D, N, r, seed, lam in all_combinations:
        experiment_counter += 1
        key = (D, N, r, seed, lam)

        if key in done:
            print(f"⏩ Skip existant: D={D}, N={N}, r={r}, lam={lam}, seed={seed}")
            continue

        print(f"\n🔹 Experiment {experiment_counter}/{len(all_combinations)}, seed {seed} "
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
        print(f"\n➡️ Solving original CFL")
        z_sol, _, captured_demand = CFL_MILP_h(instance, r)
        P0 = get_probability_a(instance, z_sol)

        # Defining Dspace
        dspace = {k: 1 for k, v in z_sol.items() if v == 0}
        if dspace:
            dspace = {list(dspace.keys())[0]: 1}
        else:
            dspace = {}

        # Lower bound with model-free case
        if lam>0:
            print(f"\n➡️ Wasserstein Lower Bound")
            _, _, _, Wbound, runtimeBound = RelativeExplanationsDistGeneric_a(
                instance, r, z_sol, alpha, dspace, distF)
        elif lam==0:
            Wbound=0
            runtimeBound="nan"

        # Feasible solution
        print(f"\n➡️ Calculating feasible solution")
        start_time_feas = time.time()
        initial_sol = feasible_solution_facility(instance, r, z_sol, alpha, dspace, captured_demand, distF, "I")
        end_time_feas = time.time()
        elapsed_time_feas = end_time_feas - start_time_feas

        # Relative Counterfactual
        print(f"\n➡️ Solving counterfactual problem")
        z_sol_new, a_new, _, captured_demand_new, _, runtimeMixedF, timersMixedF, GapMixedF = RelativeExplanationsMixedFacility(
            instance, r, z_sol, alpha, dspace, Wbound, initial_sol, lams)

        # Getting metrics
        if z_sol_new and a_new:
            instance2 = copy.deepcopy(instance)
            instance2["a"] = {(n, d): a * a_new[d] for (n, d), a in instance["a"].items()}
            P_new = get_probability_a(instance2, z_sol_new)
            Wdist, _ = WassersteinDistOpt(instance, P0, P_new, distF)

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

        # Guardar resultado parcial inmediatamente
        result_row = {
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
            "Runtime Bound": runtimeBound,
            "Runtime Feasible": elapsed_time_feas,
            "Runtime": runtimeMixedF,
            "Time Best Solution Mixed": timersMixedF.get("best_solution_time", None),
            "Time Best Bound Mixed": timersMixedF.get("last_lower_bound_time", None),
            "Gap": GapMixedF
        }

        # Adding the results
        df_prev = pd.concat([df_prev, pd.DataFrame([result_row])], ignore_index=True)
        df_prev.to_excel(output_file, index=False)
        print(f"✅ Saved in {output_file}")

    print(f"\n🎉 All experiments finished. Results in {output_file}")


if __name__ == "__main__":
    # Configuración general
    D_sizes = [10, 20]
    N_sizes = [100, 200]
    r_sizes = [4, 8]
    E_size = 5
    seeds = list(range(1, 11))
    alpha = 1
    distF = "euclidean"
    beta_d = 0.1
    lambs = [0, 0.1, 1]
    output_file = "experiments_ALL.xlsx"

    run_experiment(D_sizes, N_sizes, r_sizes, E_size, seeds, alpha, distF, beta_d, lambs, output_file)

