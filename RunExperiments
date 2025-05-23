import pandas as pd
import numpy as np
import copy
from ExplanationsCFL import *  

# General parameters
D_sizes = [10,20] 
N_sizes = [100] 
r_sizes= [4,8] 

E_size= 5
seeds = [1,2,3,4,5,6,7,8,9,10]  
alpha = 1
distF = "euclidean"
beta_d=0.1
lambs=[0,0.1,1]
results = []

n_experiments=len(D_sizes)*len(N_sizes)*len(r_sizes)*len(seeds)*len(lambs)
i=0
for N in N_sizes:
    for D in D_sizes:
        for r in r_sizes: 
            for lam in lambs:
                for seed in seeds:
            
                    lams={"L1":1,"W":lam}
                    
                    print(f"\n🔹 Experiment {i+1}/{n_experiments}, with seed {seed} (D_size={D}, N_size={N}, r_size={r}, lamW={lam} )")

                    # Generation of instance
                    ins = generate_instance(D_size=D, E_size=E_size, N_size=N, seed=seed)
                    instance= instance_coeff(ins, beta_d, np.zeros(len(ins['D'])),np.zeros(len(ins['E'])))

                    profiles = get_preference_profiles(instance, beta_d=0.5, features_d=np.zeros(len(ins['D'])),features_e=np.zeros(len(ins['E'])), num_samples=1000)
                    max_entropy_emp=max_entropy_empirical(profiles)
                    max_entropy_the=max_entropy_theorical(instance)
                    entropy_value = entropy_from_profiles(profiles)

                    #Solve CFL with Haase linearization
                    print(f"\n➡️ Solving Original CFL {i+1}/{n_experiments}, seed {seed}")
                    z_sol, w_sol, captured_demand = CFL_MILP_h(instance, r)
                    P0 = get_probability_a(instance, z_sol)

                    # Define dspace as the first d that was not selected
                    dspace = {k: 1 for k, v in z_sol.items() if v == 0}
                    dspace = {list(dspace.keys())[0]: 1}


                    # Get lower bound for Wasserstein distance
                    print(f"\n➡️ Getting lower bound W {i+1}/{n_experiments}, seed {seed}")
                    z_sol_gen, Prob_sol_gen, captured_new_gen, Wbound = RelativeExplanationsDistGeneric_a(instance, r, z_sol, alpha, dspace, distF)

                    # --- calculate feasible solution `tipo = "I"` ---
                    print(f"\n➡️ Calculating feasible solution {i+1}/{n_experiments}, seed {seed}")
                    initial_sol = feasible_solution_facility(instance, r, z_sol, alpha, dspace, captured_demand, distF, "I")
                    

                    print(f"\n➡️ Solving Counterfactual Problem {i+1}/{n_experiments}, seed {seed}")
                    z_sol_new, a_new, w_new, captured_demand_new,cost, runtimeMixedF, timersMixedF,GapMixedF = RelativeExplanationsMixedFacility(instance,r,z_sol,alpha,dspace,Wbound,initial_sol,lams)
                    

                    #Get all new results
                    Wdist = "NaN"
                    features_non_zero = {}
                    sparsity_facility = "NaN"
                    new_entropy_value = "NaN"
                    
                    if z_sol_new != [] and a_new != []:
                    # Calcule new probabilities
                        instance2 = copy.deepcopy(instance)
                        instance2["a"]={(n, d): a * a_new[d]for (n, d), a in instance["a"].items()}
                        P_new = get_probability_a(instance2, z_sol_new)
                        Wdist,_=WassersteinDist(instance,P0,P_new,distF)
                    # New features and entropy
                        new_features = retrieve_feature(instance, a_new, beta_d, "facility")
                        new_features_r = {k: round(v, 6) for k, v in new_features.items()}
                        features_non_zero = {k: v for k, v in new_features_r.items() if v != 0}
                        sparsity_facility=len(list(features_non_zero))/D
                        new_profiles = get_preference_profiles(instance, beta_d=0.5, features_d=np.array(list(new_features.values())),features_e=np.zeros(len(ins['E'])), num_samples=1000)
                        new_entropy_value = entropy_from_profiles(new_profiles)


                    # Save results
                    results.append({
                        "Exp": i + 1,
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
                        "Time Best Solution Mixed": timersMixedF["best_solution_time"],
                        "Time Best Bound Mixed": timersMixedF["last_lower_bound_time"],
                        "Gap": GapMixedF
                    })
                    i += 1

# dataframe with results
df = pd.DataFrame(results)

# excel
output_file = f"results_facility_22may.xlsx"
df.to_excel(output_file, index=False)

print(f"\n✅ Results saved in: {output_file}")
