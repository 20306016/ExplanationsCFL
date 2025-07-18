import copy
from .wasserstein_utils import WassersteinDistOpt
from .utils import get_probability_a

def feasible_solution_facility(instance, r, z_sol, alpha, dspace, captured_demand,distF,tipo):
    """
    Generate a feasible solution for the relative explanations for facility location problem 
    by selecting facilities that capture demand and adjusting parameters to ensure minimum captured demand threshold.
    
    Args:
        instance (dict): Problem instance containing data like sets D, E, N and parameters a, b, be, q.
        r (int): Number of facilities to select.
        z_sol (dict): Current solution vector for facility selection.
        alpha (float): Threshold factor for captured demand.
        dspace (dict): Facilities that must be fixed (forced to be open or closed).
        captured_demand (float): Total captured demand by initial solution.
        distF (str): Distance function type, e.g. "euclidean" or "constant" .
        tipo (str): Type of adjustment ('I' or 'II') controlling how facility adjustments are made.
        
    Returns:
        dict: A dictionary with feasible solution components:
            - 'z_feas': Binary selection vector for facilities.
            - 'a_feas': Feasible explanation.
            - 'Wupper': Wasserstein distance upper bound.
            - 'pi_feas': Transport plan from Wasserstein distance calculation.
            - 'w_feas': Auxiliary variable w.
            - 'u_feas': Auxiliary variable u.
    """

    instance_copy=copy.deepcopy(instance)
    # Extract problem sets and parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    a0=instance["a"]
    b=instance["b"]
    be=instance["be"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    

    # Initialize facility selection vector z with zeros
    z = {d: 0 for d in D}
    # Fix facilities forced by dspace (forced open or closed)
    for d in dspace.keys():
        z[d] = 1

    # Calculate captured demand per facility d based on current solution z_sol
    captured_d = {d: sum(q[n] * a0[n, d] * z_sol[d] / (sum(a0[n, d] * z_sol[d] for d in D) + b[n]) for n in N) for d in D}

    
    # Select top (r-1) facilities by captured demand to add to fixed facilities
    # Note: r-1 because some are fixed in dspace, adjust if needed
    top_indices = sorted(captured_d, key=captured_d.get,reverse=True)[0:r-1]
    
    for d in top_indices:
        z[d]=1

    
    # Internal function to compute total captured demand with new parameters a_bar (explanations)
    def new_captured(constant,tipo):
        if tipo=="I":
            # For type I, adjust a_bar to 'constant' for facilities in dspace, else 1
            a_bar={d: 1 if d not in dspace.keys() else constant for n in N for d in D}
        elif tipo=="II":
            # For type II, adjust a_bar to 'constant' for facilities in dspace and top_indices, else 1
            a_bar={d: 1 if d not in dspace.keys() and d not in top_indices else constant for n in N for d in D}
        else:
            raise ValueError("Invalid tipo argument, must be 'I' or 'II'")
        
        # Calculate sum over customers of captured demand ratio with adjusted a_bar
        return sum( sum(a_bar[ d] * a0[n, d] * z[d] for d in D) / (sum(a_bar[ d] * a0[n, d] * z[d] for d in D) + b[n]) for n in N)
    
    # Find minimal constant c such that captured demand threshold is reached
    c=1
    while True:
        if new_captured(c,tipo)>= alpha*captured_demand:
            print(f"Captured demand reached with c: {c:.4f}")
            c_sol=c
            break
        else:
            c+=0.01


    # Build explanation parameters 'a' and auxiliary dictionary aux_new_a with the scaling factor c_sol
    if tipo=="I":
        new_a={d: 1 if d not in dspace.keys()else c_sol for d in D}
        aux_new_a={(n,d):a0[n,d] if d not in dspace.keys()else c_sol*a0[n,d] for n in N for d in D}
    elif tipo=="II":
        new_a={d: 1 if d not in dspace.keys() and d not in top_indices else c_sol for d in D}
        aux_new_a={(n,d):a0[n,d] if d not in dspace.keys() and d not in top_indices else c_sol*a0[n,d] for n in N for d in D}
            
    # Create a modified instance with adjusted 'a'
    instance2=copy.deepcopy(instance_copy)
    instance2["a"]=aux_new_a
    # Calculate new probability distribution
    P_new=get_probability_a(instance2,z)
    # Calculate Wasserstein distance and optimal transport plan between original and new distributions
   
    
    Wupper,pi_values= WassersteinDistOpt(instance,get_probability_a(instance, z_sol),P_new,distF)


    # Calculate auxiliary variables w_new and u_new based on new parameters and fixed z
    w_new={(n,d): aux_new_a[n,d]*z[d]/(sum(aux_new_a[n,k]*z[k]for k in D)+b[n]) for n in N for d in D}
    u_new={(n,e): be[n,e]/(sum(aux_new_a[n,k]*z[k] for k in D)+b[n])for n in N for e in E}

    # Pack results in a dictionary and return
    initial_sol = {
        "z_feas": z,
        "a_feas": new_a,
        "Wupper": Wupper,
        "pi_feas": pi_values,
        "w_feas": w_new,
        "u_feas": u_new,
    }

    return initial_sol
