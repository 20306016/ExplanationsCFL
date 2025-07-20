import gurobipy as gp
from gurobipy import GRB
import time
from .utils import get_probability_a
from .wasserstein_utils import distFn
import ot
import numpy as np
import copy





def RelativeExplanationsDistGeneric_a(instance, r, z_sol, alpha, dspace, distF):
    """
    Computes relative counterfactual explanations considering the model-free case by optimizing a Wasserstein distance

    Args:
        instance (dict): Problem instance containing data.
        r (int): Budget or cardinality constraint for facilities.
        z_sol (dict): Current solution vector for facility locations.
        alpha (float): Threshold controlling captured demand ratio.
        dspace (dict): Dictionary with desired fixed values for some variables.
        distF (str): Distance function type ('euclidean' or 'constant').

    Returns:
        z_sol_new (dict): New facility location decisions.
        Prob_sol (dict): New probability distributions per customer.
        captured_demand (float): Total captured demand under new probabilities.
        Wdist (float): Objective Wasserstein distance value.
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C = D + E
    q = {N[i]: instance["q"][i] for i in range(len(N))}

    P0 = get_probability_a(instance, z_sol)
    epsi = 1e-5

    model = gp.Model("RelExpGen")

    # Variables
    Pd = model.addVars(N, D, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="prob_d")
    Pe = model.addVars(N, E, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="prob_e")
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    z_aux = model.addVars(N, D, vtype=GRB.BINARY)

    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")

    # Objective
    if distF == "euclidean":
        dist = distFn(instance)
        obj=gp.quicksum(pi[n, c, c_prime] * dist[c, c_prime]
                        for n in N for c in C for c_prime in C)
    elif distF == "constant":
        obj=pi.sum()
    model.setObjective(obj, GRB.MINIMIZE)


    # Constraints
    model.addConstrs(
        (gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c]
         for n in N for c in C), name="probability_constraints1")

    model.addConstrs(
        (gp.quicksum(pi[n, c, d_prime] for c in C) == Pd[n, d_prime]
         for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs(
        (gp.quicksum(pi[n, c, e] for c in C) == Pe[n, e]
         for n in N for e in E), name="probability_constraints3")

    model.addConstrs(
        (Pd.sum(n, '*') + Pe.sum(n, '*') == 1
         for n in N), name="prob_equals_1")
    
    total_captured = gp.quicksum(q[n] * Pd.sum(n, '*') for n in N)
    initial_captured = gp.quicksum(q[n] * sum(P0[n][d] for d in D) for n in N)

    model.addConstr(
        total_captured >= alpha * initial_captured,
        name="captured_demand_control")

    model.addConstrs((Pd[n, d] <= z_aux[n, d] for n in N for d in D), name="selected")

    model.addConstrs((Pd[n, d] >= epsi * z_aux[n, d] for n in N for d in D), name="selected2")

    model.addConstr(z.sum() == r, name="z_cardinality")

    model.addConstrs(z_aux[n, d] <= z[d] for n in N for d in D)

    model.addConstrs((z[d] <= z_aux.sum('*', d) for d in D), name="link_z")

    # Desired fixed values for some variables
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value

    model.setParam('TimeLimit', 3600)
    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        z_sol_new = {d: z[d].x for d in D}
        Prob_sol = {N[i]: {
            **{D[d]: Pd[N[i], D[d]].x for d in range(len(D))},
            **{E[e]: Pe[N[i], E[e]].x for e in range(len(E))}
        } for i in range(len(N))}
        captured_demand = sum(q[n] * sum(Pd[n, d].x for d in D) for n in N)
        Wdist = model.ObjVal
        runtime=model.Runtime

    else:
        z_sol_new = []
        Prob_sol = []
        captured_demand = []
        Wdist = 0
        runtime="nan"

    return z_sol_new, Prob_sol, captured_demand, Wdist, runtime




def RelativeExplanationsMixedFacility(instance,r,z_sol,alpha,dspace,Wbound,initial_sol,lams):
    """
    Optimization problem for calculating relative explanations 
    for context-dependent choice-based competitive facility problem
    
    Parameters:
    - instance: dict, problem instance data including sets, parameters.
    - r: int, number of facilities to select.
    - z_sol: dict, initial binary solution for facilities.
    - alpha: float, minimum fraction of demand to be captured.
    - dspace: dict, facilities fixed to 0 or 1 in solution {facility: 0 or 1}.
    - Wbound: float, lower bound for Wasserstein distance in regularization.
    - initial_sol: dict, initial feasible solution to warm start optimization (optional).
    - lams: dict, regularization weights {'L1': float, 'W': float}.
    
    Returns:
    - z_sol_new: dict, new facility selection (binary variables).
    - a_new: dict, new facility-context (explanations).
    - w_sol: dict, new assignment probabilities for customers to facilities.
    - captured_demand: float, total weighted captured demand in solution.
    - objective: float, final objective function value.
    - total_runtime: float, optimization runtime in seconds.
    - timers: dict, timing info for best solutions and bounds.
    - gap: float, final MIP gap.
    """
    # Extract sets and parameters from instance
    D = instance["D"]        # Facilities
    E = instance["E"]        # Competitive facilities
    N = instance["N"]        # Customers
    C = D + E                # All facilities
    q = {N[i]: instance["q"][i] for i in range(len(N))}  # Demand weights
    a0 = instance["a"]       # Original utility coefficients
    b = instance["b"]        # Demand offsets
    be = instance["be"]      # External demand parameters
    laml1 = lams['L1']       # L1 regularization weight
    lamW = lams["W"]         # Wasserstein regularization weight
   
    # Compute original assignment probabilities given current solution
    P0 = get_probability_a(instance, z_sol)

    #Compute distances
    dist = distFn(instance)
    
    # Initialize Gurobi model
    model = gp.Model("RelExp")
    
    # Decision variables
    a = model.addVars(D, vtype=GRB.CONTINUOUS, lb=1, name="a")  # new contextual variables 
    aux_a = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="a_aux")  # auxiliary vars for |1 - a|
    z = model.addVars(D, vtype=GRB.BINARY, name="z")              # facility selection
    w = model.addVars(N, D, vtype=GRB.CONTINUOUS, lb=0, name="w")  # assignment probs customer->facility
    u = model.addVars(N, E, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="u")  # assignment probs customer->competitive
    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")  # transport plan variables
    objvar = model.addVar(vtype=GRB.CONTINUOUS, lb=lamW*Wbound)   # objective value

    # Warm start if initial solution provided
    if initial_sol:
        for n in N:
            for d in D:
                a[d].start = initial_sol["a_feas"][d]
                w[n,d].start = initial_sol["w_feas"][(n,d)]
                z[d].start = initial_sol["z_feas"][d]
                for e in E:
                    u[n,e].start = initial_sol['u_feas'][(n,e)]
        for n in N:
            for c in C:
                for c_prime in C:
                    pi[n, c, c_prime].start = initial_sol['pi_feas'][(n, c, c_prime)]

    # Objective: weighted sum of L1 norm of (1 - a) plus normalized Wasserstein distance
    model.setObjective(objvar, GRB.MINIMIZE)
    
    model.addConstr(
        objvar == 
        laml1 * aux_a.sum() + 
        (lamW / Wbound) * gp.quicksum(pi[n, c, cp] * dist[c, cp] for n in N for c in C for cp in C)
    )
    

    # Demand capture constraint (weighted sum of assignment probabilities)
    model.addConstr(
        gp.quicksum(q[n] * w.sum(n, '*') for n in N) 
        >= alpha * gp.quicksum(q[n] * sum(P0[n][d] for d in D) for n in N),
        name="captured_demand"
    )
    # Transport plan constraints: marginal probabilities equal P0 and assignment variables
    
    model.addConstrs(
        (pi.sum(n, c, '*') == P0[n][c] for n in N for c in C),
        name="pi_marg1"
    )


    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) == w[n,d_prime]
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) == u[n,e]
                      for n in N for e in E), name="probability_constraints3")
    # Assignment probabilities sum to at most 1 for each customer
    model.addConstrs(
        (w.sum(n, '*') + u.sum(n, '*') <= 1 for n in N),
        name="assign_total"
    )
    
    # Linking constraints between w, a, z, b variables for feasibility
    model.addConstrs (a0[n,d]*a[d]*(w[n,d]-z[d])+b[n]*w[n,d]<=0 
             for n in N for d in D)
    
    # Linking constraints involving u, w, a, be and z
    model.addConstrs((w[n,d]-a0[n,d]*a[d]/be[n,e]*u[n,e]<=0 for n in N for d in D for e in E))
    model.addConstrs((a0[n,d]*a[d]/be[n,e]*u[n,e]<=w[n,d]+(1-z[d]) for n in N for d in D for e in E))
    

    # Cardinality constraint: exactly r facilities selected
    model.addConstr(z.sum() == r, name="cardinality")
    
    # Auxiliary variables constraints for L1 norm calculation
    model.addConstrs(aux_a[d]>=(1-a[d]) for d in D)
    model.addConstrs(aux_a[d]>=-(1-a[d])  for d in D)

    # Fix facilities according to dspace (forcing 0 or 1)
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    # Setup timing and callback
    timers = {
        "start_time": time.time(),
        "best_solution_time": None,
        "last_lower_bound_time": None,
          }
    def callback_with_data(model, where):
        my_callback(model, where, timers)

    # Time limit
    model.setParam('TimeLimit', 3600)
    model.optimize(callback=callback_with_data)
    gap=model.MIPGap
    
    # Check for solution availability
    if model.status == gp.GRB.OPTIMAL or  model.SolCount > 0:
        z_sol_new={d: z[d].x for d in D}
        a_new={d: a[ d].X for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=sum(q[n] * sum(w_sol[n, d] for d in D) for n in N)
        objective=model.ObjVal
        total_runtime = model.Runtime
        gap=model.MIPGap
        

    else:
        z_sol_new=[]
        a_new=[]
        w_sol=[]
        captured_demand=[]
        objective=[]
        total_runtime = []
        gap=[]

    return z_sol_new, a_new, w_sol,captured_demand,objective, total_runtime,timers,gap





def my_callback(model, where, timers):
    """
    Custom Gurobi callback to track best objective values and bounds over time.

    Args:
        model (gurobipy.Model): The Gurobi model instance.
        where (int): Callback location identifier.
        timers (dict): Dictionary to store timing and bound information.
            Expected keys:
                - "start_time": float, timestamp when optimization started.
                - "last_upper_bound": float, best known upper bound so far.
                - "best_solution_time": float, time when best solution was found.
                - "last_lower_bound": float, best known lower bound so far.
                - "last_lower_bound_time": float, time when best lower bound was updated.

    Functionality:
        - On finding a new solution (MIPSOL), updates upper bound and records time.
        - Tracks lower bound updates both on MIPSOL and during MIP progress.
    """

    current_time = time.time() - timers["start_time"]
    
    if where == gp.GRB.Callback.MIPSOL:
	 # A new integer feasible solution was found
        objval = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        bound= model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
        
	# Update best upper bound and time if improved
        if "last_upper_bound" not in timers or objval < timers["last_upper_bound"]:
            timers["last_upper_bound"] = objval
            timers["best_solution_time"] = current_time
        # Update best lower bound and time if improved
        if "last_lower_bound" not in timers or bound > timers["last_lower_bound"]:
            timers["last_lower_bound"] = bound
            timers["last_lower_bound_time"] = current_time

        elif where == gp.GRB.Callback.MIP:
            print("Entrando en MIP callback")
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
				
            # Update best lower bound and time if improved
            if "last_lower_bound" not in timers or bound > timers["last_lower_bound"]:
                timers["last_lower_bound"] = bound
                timers["last_lower_bound_time"] = current_time

