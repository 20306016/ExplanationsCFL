import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def generate_instance(D_size, E_size, N_size, width=20, seed=0):


    """
    Generate an instance.
    
    Args:
        D_size (int): Number of available locations
        E_size (int): Number of competitive facilities
        N_size (int): Number of customers
        width (int): width of the square on which customers are uniformly distributed
        seed (int): seed provided to the random number generator

    Returns:
        dict: Diccionario con conjuntos y parámetros generados.
    """
    if seed != 0:
        np.random.seed(seed)

    # Generate random coordinates for facilities and customers
    coord_E = width * np.random.rand(E_size, 2)
    coord_D = width * np.random.rand(D_size, 2)
    coord_N = width * np.random.rand(N_size, 2)

    # Distances 
    theta_d = np.zeros((N_size, D_size))  # L1 distance from customers to available locations
    theta_e = np.zeros((N_size, E_size))  # L1 distance from customers to competitive facilities

    for n in range(N_size):
        for d in range(D_size):
            theta_d[n, d] = np.sum(np.abs(coord_N[n] - coord_D[d]))
        for e in range(E_size):
            theta_e[n, e] = np.sum(np.abs(coord_N[n] - coord_E[e]))

    theta_d = np.round(theta_d, decimals=3)
    theta_e = np.round(theta_e, decimals=3)

    # Weight of customers
    q = np.ones(N_size, dtype=int) #qn=1 

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
    plt.figure(figsize=(6, 6))
    
    # All the locations
    pos_D = {f"Facility_D_{facility}": pos for facility, pos in zip(instance['D'], instance['coord_D'])}
    pos_E = {f"Facility_E_{facility}": pos for facility, pos in zip(instance['E'], instance['coord_E'])}
    pos_N = {f"Customer_{customer}": pos for customer, pos in zip(instance['N'], instance['coord_N'])}
    
    # Available locations 
    for node, coords in pos_D.items():
        facility_id = node.split('_')[-1]  # Get 'd1', 'd2', etc.
        if facility_status and facility_status.get(facility_id) == 1.0:
            color = 'black'  # Color black if the status is 1.0
        else:
            color = 'white'  # Otherwise, color white
        plt.scatter(*coords, color=color, edgecolor='black', s=200, marker='^') 
    
    # Competitive facilities
    for node, coords in pos_E.items():
        plt.scatter(*coords, color='black', s=200, marker='D')
    
    # Customers
    for node, coords in pos_N.items():
        plt.scatter(*coords, color='black', s=100, marker='o')  
        

    for node, coords in {**pos_D, **pos_E, **pos_N}.items():
        plt.text(coords[0], coords[1], node, fontsize=10, ha='center', color='black')
    
    # Title
    plt.title("Instance visualizations")
    plt.axis('equal') 
    plt.show()


def utility_function(theta, n, c, beta_d,beta_a):
    """
    Defines the utility function for a given customer and location.

    Args:
        theta (ndarray): Distance matrix (either theta_d or theta_e).
        c (int): Index of the location.
        n (int): Index of the customer.
        x (list): Parameter indicating additional amenities for customers (for example)
        beta_d (float): Coefficient impacting the distance utility
        beta_a (float): Coefficient (variable) indicating if there are amenities or not 

    Returns:
        float: Computed utility value.
    """
    return -(beta_d-beta_a[n,c]) * theta[n, c]


def CFL_MILP(instance, r, beta_a, beta_d=0.1):
    """
    Solves the optimization problem using Gurobi.

    Args:
        instance (dict): Instance generated
        x_values (list): Binary parameter values for x (change to something generally)
        beta (float): Coefficient beta for the utility function.
        r (integer) : budget (number of facilities to locate)
    """
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    theta_d = instance["theta_d"]
    theta_e = instance["theta_e"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}

    # Create the model
    model = gp.Model("CFL_MILP")

    # Variables
    z = model.addVars(D, vtype=gp.GRB.BINARY, name="z")  # Binary variables for the location of the facilities
    w = model.addVars(N, D, vtype=gp.GRB.CONTINUOUS, lb=0, name="w")  # Continuous auxiliary variables

    # Compute a_{nd} and b_n using the utility function
    a = {(N[n], D[d]): np.exp(utility_function(theta_d, n, d, beta_d,beta_a)) for n in range(len(N)) for d in range(len(D))}
    b = {N[n]: sum(np.exp(utility_function(theta_e, n, e, beta_d,beta_a)) for e in range(len(E))) for n in range(len(N))}

    # Objective function
    obj = gp.quicksum(q[n] * (gp.quicksum(a[n, d] / b[n] * z[d] for d in D) -
                          gp.quicksum(a[n, d] / b[n] * w[n, d] for d in D)) for n in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    model.addConstr(gp.quicksum(z[d] for d in D) == r, "Budget")

    model.addConstrs(
        w[n, d] >= gp.quicksum(a[n, k] / b[n] * z[k] for k in D) -
                    gp.quicksum(a[n, k] / b[n] * w[n, k] for k in D) - (1 - z[d])
        for n in N for d in D
    )
    model.addConstrs( 
        w[n,d]<=z[d] for d in D for n in N
    )
    model.addConstrs(
        w[n,d]<=gp.quicksum(a[n, k] / b[n] * z[k] for k in D) -
                    gp.quicksum(a[n, k] / b[n] * w[n, k] for k in D)
        for n in N for d in D
    )

    # Solve the model
    model.optimize()

    # Display results
    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol={d: z[d].x for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=gp.quicksum(q[n] *( gp.quicksum(a[n, d] * z_sol[d] / b[n] for d in D) - 
                                       gp.quicksum(a[n,d]*w_sol[n,d]/b[n] for d in D)) for n in N) 
        return z_sol, captured_demand
        #print("w:", {(n, d): w[n, d].x for n in N for d in D if w[n, d].x > 1e-6})
    else:
        print("No optimal solution found.")


def get_probability(instance, z_sol, beta_a, beta_d):
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    theta_d = instance["theta_d"]
    theta_e = instance["theta_e"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    a = {(N[n], D[d]): np.exp(utility_function(theta_d, n, d, beta_d,beta_a)) for n in range(len(N)) for d in range(len(D))}
    b = {N[n]: sum(np.exp(utility_function(theta_e, n, e, beta_d,beta_a)) for e in range(len(E))) for n in range(len(N))}
    be= {(N[n], E[e]): np.exp(utility_function(theta_e, n, e, beta_d, beta_a)) for n in range(len(N)) for e in range (len(E))}
    
    Prob = { N[i]: {
        **{D[d]: (a[N[i], D[d]] * z_sol[D[d]]) / (sum(a[N[i], D[k]] * z_sol[D[k]] for k in range(len(D))) + b[N[i]])
            for d in range(len(D))},
        **{E[e]: (be[N[i],E[e]] / (sum(a[N[i], D[k]] * z_sol[D[k]] for k in range(len(D))) + b[N[i]]))
            for e in range(len(E)) } 
            } for i in range(len(N))}
    
    return Prob


def distFn(instance):
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
    Solves the optimization problem to get the Wasserstein distance between two discrete distributions form n customers

    Args:
        Prob1 (dict): First distribution for the n customers
        Prob2 (dict): Second distribution for the n customers
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


def RelativeExplanationsDist(instance,r,beta_a, beta_d,z_sol,alpha,dspace,Wbound):
    
    # Parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C=D+E
    theta_d = instance["theta_d"]
    theta_e = instance["theta_e"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    a0 = {(N[n], D[d]): np.exp(utility_function(theta_d, n, d, beta_d,beta_a)) for n in range(len(N)) for d in range(len(D))}
    b = {N[n]: sum(np.exp(utility_function(theta_e, n, e, beta_d,beta_a)) for e in range(len(E))) for n in range(len(N))}
    be= {(N[n], E[e]): np.exp(utility_function(theta_e, n, e, beta_d, beta_a)) for n in range(len(N)) for e in range (len(E))}
    P0 = get_probability(instance, z_sol, beta_a,beta_d)
    M = 300
    
    # Model
    model = gp.Model("RelExp")
    
    # Variables
    a = model.addVars(N, D, vtype=GRB.CONTINUOUS, name="a",ub=5)
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    w = model.addVars(N, D, vtype=GRB.CONTINUOUS, name="w")
    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")
    #gurobi does not support product of three continuos variables, so we use this auxilary variable
    u = model.addVars(N,D, vtype= GRB.CONTINUOUS, name="u") 

    objvar=model.addVar(vtype=GRB.CONTINUOUS, lb=Wbound)
    
    #Objective
    model.setObjective(objvar, GRB.MINIMIZE)
    
    # Constraints

    model.addConstr(objvar==gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C))
    
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c] 
                     for n in N for c in C), name="probability_constraints1")

   
    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) ==
                      (a[n, d_prime] * z[d_prime]) / b[n] -
                      (a[n, d_prime] / b[n]) * gp.quicksum(u[n,k] for k in D)
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) ==
                      be[n,e] / b[n] -
                      (be[n,e] / b[n]) * gp.quicksum(a[n, k] * w[n, k] for k in D)
                      for n in N for e in E), name="probability_constraints3")

    model.addConstr(
        gp.quicksum(q[n] * gp.quicksum((a[n, d] * z[d]) / b[n] - (a[n, d] / b[n]) *
                                       gp.quicksum(u[n,k] for k in D) for d in D) for n in N) 
        >= alpha * gp.quicksum(q[n] * gp.quicksum(P0[n][d] for d in D) for n in N), name="captured_demand_control")
    
    model.addConstrs((w[n, d] >= 1/b[n] - gp.quicksum((a[n, k] / b[n]) * w[n, k] for k in D) - M * (1 - z[d])
                     for n in N for d in D), name="w_lower_bound")
    
    model.addConstrs((w[n, d] <= M * z[d] for n in N for d in D), name="w_upper_bound_1")
    
    model.addConstrs((w[n, d] <= 1/b[n] - gp.quicksum((a[n, k] / b[n]) * w[n, k] for k in D)
                     for n in N for d in D), name="w_upper_bound_2")

    model.addConstrs( u[n,d] == a[n,d]*w[n,d] for n in N for d in D)
    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, name="z_cardinality")
    
    # define D
    #model.addConstr(z['d1']==1, name="desired space")
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    # Spolve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol_new={d: z[d].x for d in D}
        a_new={(n, d): a[n, d].X for n in N for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=sum(q[n]* sum(a_new[n,d]* z_sol_new[d] / 
                                                      (sum(a_new[n,k] * z_sol_new[k] for k in D) + b[n]) for d in D) for n in N)
        Wdist=model.ObjVal
        return z_sol_new, a_new, captured_demand, Wdist

    else:
        z_sol_new={d: z[d].x for d in D}
        a_new={(n, d): a[n, d].X for n in N for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D} 
        captured_demand=sum(q[n]* sum(a_new[n,d]* z_sol_new[d] / 
                                                      (sum(a_new[n,k] * z_sol_new[k] for k in D) + b[n]) for d in D) for n in N)
        Wdist=model.ObjVal
        print("No optimal solution found.")
        return z_sol_new, a_new, captured_demand,Wdist


def RelativeExplanationsDistGeneric(instance,r,beta_a, beta_d,z_sol,alpha,dspace):
    # Parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C=D+E
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    P0 = get_probability(instance, z_sol, beta_a,beta_d)
    M = 250
    epsi=10e-6
    
    # Model
    model = gp.Model("RelExpGen")
    
    # Variables
    Pd=model.addVars(N,D, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="prob_d")
    Pe=model.addVars(N,E,vtype=GRB.CONTINUOUS, lb=0,ub=1, name="prob_e")
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    z_aux=model.addVars(N,D, vtype=GRB.BINARY)
    
    

    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")
    
    #Objective
    model.setObjective(gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C), GRB.MINIMIZE)
    
    # Constraints
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c]
                     for n in N for c in C), name="probability_constraints1")

   
    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) == Pd[n,d_prime]
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) == Pe[n,e]
                      for n in N for e in E), name="probability_constraints3")


    model.addConstrs((gp.quicksum(Pd[n,d] for d in D)+gp.quicksum(Pe[n,e] for e in E) == 1
                      for n in N), name="prob_equals_1")
    
    model.addConstr(gp.quicksum(q[n] * gp.quicksum(Pd[n,d] for d in D) for n in N)
                    >= alpha * gp.quicksum(q[n] * gp.quicksum(P0[n][d] for d in D) for n in N), name="captured_demand_control")

    model.addConstrs((Pd[n,d]<= z_aux[n,d] for n in N for d in D), name="selected")
    
    model.addConstrs((Pd[n,d]>=epsi*z_aux[n,d] for n in N for d in D), name="selected2")
    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, name="z_cardinality")

    #model.addConstr(z['d1']==1, name="desired_space")

    model.addConstrs(z_aux[n,d]<=z[d] for n in N for d in D)

    model.addConstrs(z[d]<= gp.quicksum(z_aux[n,d] for n in N) for d in D)

    #desiredspace
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    
    # Spolve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol_new={d: z[d].x for d in D}

        
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}

       
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        return z_sol_new, Prob_sol, captured_demand
    else:
        print("No optimal solution found.")
        z_sol_new={d: z[d].x for d in D}

        
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        return z_sol_new, Prob_sol, captured_demand


#same function but for when i hav directly the values of the costs and not the distances 


def read_data(filename):
    with open(filename, "r") as file:
    # Leer la primera línea (parámetros generales)
        first_line = list(map(int, file.readline().strip().split()))
        num_clients, num_facilities, max_facilities = first_line

        # Leer la segunda línea (demandas de cada cliente)
        demands = np.array(list(map(float, file.readline().strip().split())))

        # Leer la tercera línea (coste del cliente en la ubicación actual)
        incumbent_costs = np.array(list(map(float, file.readline().strip().split())))

        # Leer los costes de cada cliente a cada ubicación (matriz de MxN)
        cost_matrix = np.loadtxt(file) 

    a_dict = { (f"n{i+1}", f"d{j+1}"): np.exp(cost_matrix[j, i]) 
              for j in range(num_facilities) 
              for i in range(num_clients) }
    b_dict= {f"n{i+1}": np.exp(incumbent_costs[i]) for i in range(num_clients)}
    be_dict={ (f"n{i+1}","e1"): np.exp(incumbent_costs[i]) for i in range(num_clients)}

    D = [f"d{i+1}" for i in range(cost_matrix.shape[0])]
    E = [f"e{i+1}" for i in range(1)]
    N = [f"n{i+1}" for i in range(num_clients)]

    return {
        "D": D,
        "E": E,
        "N": N,
        "a": a_dict,
        "b" : b_dict,
        "be": be_dict,
        "q": demands

    }



def CFL_MILP_a(instance, r ):
    """
    Solves the optimization problem using Gurobi.

    Args:
        instance (dict): Instance generated
        x_values (list): Binary parameter values for x (change to something generally)
        beta (float): Coefficient beta for the utility function.
        r (integer) : budget (number of facilities to locate)
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
    z = model.addVars(D, vtype=gp.GRB.BINARY, name="z")  # Binary variables for the location of the facilities
    w = model.addVars(N, D, vtype=gp.GRB.CONTINUOUS, lb=0, name="w")  # Continuous auxiliary variables

    # Objective function
    obj = gp.quicksum(q[n] * (gp.quicksum(a[n, d] / b[n] * z[d] for d in D) -
                          gp.quicksum(a[n, d] / b[n] * w[n, d] for d in D)) for n in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    model.addConstr(gp.quicksum(z[d] for d in D) == r, "Budget")

    model.addConstrs(
        w[n, d] >= gp.quicksum(a[n, k] / b[n] * z[k] for k in D) -
                    gp.quicksum(a[n, k] / b[n] * w[n, k] for k in D) - (1 - z[d])
        for n in N for d in D
    )
    model.addConstrs( 
        w[n,d]<=z[d] for d in D for n in N
    )
    model.addConstrs(
        w[n,d]<=gp.quicksum(a[n, k] / b[n] * z[k] for k in D) -
                    gp.quicksum(a[n, k] / b[n] * w[n, k] for k in D)
        for n in N for d in D
    )

    # Solve the model
    model.optimize()

    # Display results
    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol={d: z[d].x for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=gp.quicksum(q[n] *( gp.quicksum(a[n, d] * z_sol[d] / b[n] for d in D) - 
                                       gp.quicksum(a[n,d]*w_sol[n,d]/b[n] for d in D)) for n in N) 
        return z_sol, captured_demand
        #print("w:", {(n, d): w[n, d].x for n in N for d in D if w[n, d].x > 1e-6})
    else:
        print("No optimal solution found.")
        z_sol={d: z[d].x for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=gp.quicksum(q[n] *( gp.quicksum(a[n, d] * z_sol[d] / b[n] for d in D) - 
                                       gp.quicksum(a[n,d]*w_sol[n,d]/b[n] for d in D)) for n in N) 
        return z_sol, captured_demand
    

def get_probability_a(instance, z_sol):
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


def RelativeExplanationsDist_a(instance,r,z_sol,alpha,dspace,Wbound,distF):
    
    # Parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C=D+E
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    a0=instance["a"]
    b=instance["b"]
    be=instance["be"]
   
    P0 = get_probability_a(instance, z_sol)
    M = 10000
    
    # Model
    model = gp.Model("RelExp")
    
    # Variables
    a = model.addVars(N, D, vtype=GRB.CONTINUOUS, name="a")
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    w = model.addVars(N, D, vtype=GRB.CONTINUOUS, name="w")
    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")
    #gurobi does not support product of three continuos variables, so we use this auxilary variable
    u = model.addVars(N,D, vtype= GRB.CONTINUOUS, name="u") 

    objvar=model.addVar(vtype=GRB.CONTINUOUS, lb=Wbound)
    
    #Objective
    model.setObjective(objvar, GRB.MINIMIZE)
    
    # Constraints
    if distF=="euclidean":
        model.addConstr(objvar==gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C))
    if distF=="constant":
        model.addConstr(objvar==gp.quicksum(pi[n, c, c_prime] 
                                    for n in N for c in C for c_prime in C))
    
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c] 
                     for n in N for c in C), name="probability_constraints1")

   
    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) ==
                      (a[n, d_prime] * z[d_prime]) / b[n] -
                      (a[n, d_prime] / b[n]) * gp.quicksum(u[n,k] for k in D)
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) ==
                      be[n,e] / b[n] -
                      (be[n,e] / b[n]) * gp.quicksum(a[n, k] * w[n, k] for k in D)
                      for n in N for e in E), name="probability_constraints3")

    model.addConstr(
        gp.quicksum(q[n] * gp.quicksum((a[n, d] * z[d]) / b[n] - (a[n, d] / b[n]) *
                                       gp.quicksum(u[n,k] for k in D) for d in D) for n in N) 
        >= alpha * gp.quicksum(q[n] * gp.quicksum(P0[n][d] for d in D) for n in N), name="captured_demand_control")
    
    model.addConstrs((w[n, d] >= 1/b[n] - gp.quicksum((a[n, k] / b[n]) * w[n, k] for k in D) - M * (1 - z[d])
                     for n in N for d in D), name="w_lower_bound")
    
    model.addConstrs((w[n, d] <= M * z[d] for n in N for d in D), name="w_upper_bound_1")
    
    model.addConstrs((w[n, d] <= 1/b[n] - gp.quicksum((a[n, k] / b[n]) * w[n, k] for k in D)
                     for n in N for d in D), name="w_upper_bound_2")

    model.addConstrs( u[n,d] == a[n,d]*w[n,d] for n in N for d in D)
    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, name="z_cardinality")
    
    # define D
    #model.addConstr(z['d1']==1, name="desired space")
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    # Spolve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol_new={d: z[d].x for d in D}
        a_new={(n, d): a[n, d].X for n in N for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=sum(q[n]* sum(a_new[n,d]* z_sol_new[d] / 
                                                      (sum(a_new[n,k] * z_sol_new[k] for k in D) + b[n]) for d in D) for n in N)
        Wdist=model.ObjVal
        return z_sol_new, a_new, captured_demand, Wdist

    else:
        z_sol_new={d: z[d].x for d in D}
        a_new={(n, d): a[n, d].X for n in N for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D} 
        captured_demand=sum(q[n]* sum(a_new[n,d]* z_sol_new[d] / 
                                                      (sum(a_new[n,k] * z_sol_new[k] for k in D) + b[n]) for d in D) for n in N)
        Wdist=model.ObjVal
        print("No optimal solution found.")
        return z_sol_new, a_new, captured_demand,Wdist


def RelativeExplanationsDistGeneric_a(instance,r,z_sol,alpha,dspace,distF):
    # Parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C=D+E
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    a0=instance["a"]
    b=instance["b"]
    be=instance["be"]
    P0 = get_probability_a(instance, z_sol)
    epsi=10e-6
    
    # Model
    model = gp.Model("RelExpGen")
    
    # Variables
    Pd=model.addVars(N,D, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="prob_d")
    Pe=model.addVars(N,E,vtype=GRB.CONTINUOUS, lb=0,ub=1, name="prob_e")
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    z_aux=model.addVars(N,D, vtype=GRB.BINARY)
    
    

    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")
    
    #Objective
    if distF=="euclidean":
        model.setObjective(gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C), GRB.MINIMIZE)
    if distF=="constant":
        model.setObjective(gp.quicksum(pi[n, c, c_prime] 
                                    for n in N for c in C for c_prime in C), GRB.MINIMIZE)
    
    # Constraints
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c]
                     for n in N for c in C), name="probability_constraints1")

   
    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) == Pd[n,d_prime]
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) == Pe[n,e]
                      for n in N for e in E), name="probability_constraints3")


    model.addConstrs((gp.quicksum(Pd[n,d] for d in D)+gp.quicksum(Pe[n,e] for e in E) == 1
                      for n in N), name="prob_equals_1")
    
    model.addConstr(gp.quicksum(q[n] * gp.quicksum(Pd[n,d] for d in D) for n in N)
                    >= alpha * gp.quicksum(q[n] * gp.quicksum(P0[n][d] for d in D) for n in N), name="captured_demand_control")

    model.addConstrs((Pd[n,d]<= z_aux[n,d] for n in N for d in D), name="selected")
    
    model.addConstrs((Pd[n,d]>=epsi*z_aux[n,d] for n in N for d in D), name="selected2")
    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, name="z_cardinality")

    #model.addConstr(z['d1']==1, name="desired_space")

    model.addConstrs(z_aux[n,d]<=z[d] for n in N for d in D)

    model.addConstrs(z[d]<= gp.quicksum(z_aux[n,d] for n in N) for d in D)

    #desiredspace
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    
    # Spolve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print("\nOptimal Solution:")
        print("z:", {d: z[d].x for d in D})
        z_sol_new={d: z[d].x for d in D}

        
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}

       
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        return z_sol_new, Prob_sol, captured_demand
    else:
        print("No optimal solution found.")
        z_sol_new={d: z[d].x for d in D}

        
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        return z_sol_new, Prob_sol, captured_demand


