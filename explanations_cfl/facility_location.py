import gurobipy as gp
from gurobipy import GRB


def CFL_MILP_h(instance, r ):
    """
    Solve a capacitated facility location problem using a MILP formulation in Gurobi.

    Args:
        instance (dict): Dictionary containing the instance data with the following keys:
            - "N" (list): List of demand points.
            - "D" (list): List of candidate facility locations.
            - "E" (unused): List of competitive facilities.
            - "a" (dict): Dictionary mapping (n, d) pairs customer-to-candidate utility exponentials.
            - "b" (dict): Dictionary mapping (n) to customer-to-allcomeptitors utility exponentials.
            - "q" (list): List of demand values corresponding to demand points in N.
        r (int): Number of facilities to locate (budget).

    Returns:
        tuple:
            - z_sol (dict): Dictionary with keys in D and binary values indicating opened facilities.
            - w_sol (dict): Dictionary mapping (n, d) pairs to the amount of demand from `n` assigned to facility `d`.
            - captured_demand (float): Total demand captured by the selected facilities.

    Notes:
        - The model is solved with a time limit of 3600 seconds.
        - If no feasible solution is found, empty structures are returned.
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    a=instance["a"]
    b=instance["b"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}

    # Create the model
    model = gp.Model("CFL_MILP")

    # Variables
    z = model.addVars(D, vtype=gp.GRB.BINARY, name="z")  
    w = model.addVars(N, D, vtype=gp.GRB.CONTINUOUS, lb=0, name="w")  
    w2 = model.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, name="w2")

    # Objective function
    obj = gp.quicksum(q[n] * w.sum(n, '*') for n in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    
    model.addConstrs(
        (w2[n] + w.sum(n, '*') <= 1 for n in N),
        name="total_demand"
    )

    model.addConstrs(
            a[n,d]*(w[n, d]-z[d])+b[n]*w[n,d] <= 0
            for n in N for d in D)
    
    model.addConstrs((w[n, d] - a[n,d]/b[n]*w2[n]<= 0 for n in N for d in D), name="w_upper_bound_1")
    
    model.addConstr(z.sum() == r, name="budget")


    # Solve the model
    model.setParam('TimeLimit', 3600)
    model.optimize()

    # Display results
    if model.status == gp.GRB.OPTIMAL or model.SolCount > 0:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol={d: z[d].x for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=sum(q[n] *( sum(w_sol[n,d] for d in D)) for n in N) 
    else:
        print("No optimal solution found.")
        z_sol=[]
        w_sol=[]
        captured_demand=[]
    return z_sol,w_sol, captured_demand
    
