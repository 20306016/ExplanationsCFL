import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import heapq
import copy
import time
from collections import Counter


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
    np.random.seed(seed)

    # Generate random coordinates for facilities and customers
    coord_E = width * np.random.rand(E_size, 2)
    coord_D = width * np.random.rand(D_size, 2)
    coord_N = width * np.random.rand(N_size, 2)

    # Distances 
    theta_d = np.abs(coord_N[:, np.newaxis] - coord_D).sum(axis=2) # L1 distance from customers to available locations
    theta_e = np.abs(coord_N[:, np.newaxis] - coord_E).sum(axis=2) # L1 distance from customers to competitive facilities

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
    D=len(instance["D"])
    N=len(instance["N"])
    E=len(instance["E"])

    plt.figure(figsize=(6, 6))
    
    # All the locations
    pos_D = {f"{facility}": pos for facility, pos in zip(instance['D'], instance['coord_D'])}
    pos_E = {f"{facility}": pos for facility, pos in zip(instance['E'], instance['coord_E'])}
    pos_N = {f"{customer}": pos for customer, pos in zip(instance['N'], instance['coord_N'])}
    
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
        plt.scatter(*coords, color='black', s=80, marker='o')  
        

    for node, coords in {**pos_D, **pos_E, **pos_N}.items():
        if not node.startswith("Customer_"):
            plt.text(coords[0], coords[1]+0.5, node, fontsize=10, ha='center', color='black')
    
    # Title
    plt.title("Instance visualizations")
    plt.axis('equal') 
    plt.savefig("instance_D_"+str(D)+"E_"+str(E)+"N_"+str(N)+".png", dpi=300, bbox_inches='tight')
    plt.show()


def utility_function(theta, n, c, beta_d,features):
    """
    Defines the utility function for a given customer and location.

    Args:
        theta (ndarray): Distance matrix (either theta_d or theta_e).
        c (int): Index of the location.
        n (int): Index of the customer.
        x (list): Parameter indicating additional amenities for customers (for example)
        beta_d (float): Coefficient impacting the distance utility
        features (matrix): feature[n,c]: pricing from customer n to facility c

    Returns:
        float: Computed utility value.
    """
    return -beta_d * theta[n, c] + features[c]


def instance_coeff(instance, beta_d,features_d, features_e):
    instance2=copy.deepcopy(instance)
    instance2["a"] = {(instance["N"][n], instance["D"][d]): np.exp(utility_function(instance["theta_d"], n, d, beta_d,features_d)) for n in range(len(instance["N"])) for d in range(len(instance["D"]))}
    instance2["b"]= {instance["N"][n]: sum(np.exp(utility_function(instance["theta_e"], n, e, beta_d,features_e)) for e in range(len(instance["E"]))) for n in range(len(instance["N"]))}
    instance2["be"]= {(instance["N"][n], instance["E"][e]): np.exp(utility_function(instance["theta_e"], n, e, beta_d, features_e)) for n in range(len(instance["N"])) for e in range (len(instance["E"]))}

    return instance2



def get_preference_profiles(instance, beta_d, features_d,features_e, num_samples=1000, seed=0):
    np.random.seed(seed)
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    theta_d = instance["theta_d"]
    theta_e = instance["theta_e"]

    profiles = []

    for _ in range(num_samples):
        eps_d = np.random.gumbel(size=(len(N), len(D)))
        eps_e = np.random.gumbel(size=(len(N), len(E)))

        for n in range(len(N)):
            u_d = [utility_function(theta_d, n, d, beta_d, features_d) + eps_d[n, d] for d in range(len(D))]
            u_e = [utility_function(theta_e, n, e, beta_d, features_e) + eps_e[n, e] for e in range(len(E))]

            best_competitor = max(u_e)
            preferred_d_indices = [d for d, u in enumerate(u_d) if u > best_competitor]

            profile = tuple(1 if d in preferred_d_indices else 0 for d in range(len(D)))
            profiles.append(profile)

    return profiles

def entropy_from_profiles(profiles):
    count = Counter(profiles)
    total = sum(count.values())
    probs = np.array([v / total for v in count.values()])
    return -np.sum(probs * np.log(probs))

def max_entropy_empirical(profiles):
    num_profiles = len(set(profiles))  # número de perfiles únicos
    return np.log(num_profiles) if num_profiles > 0 else 0.0

def max_entropy_theorical(instance):
    N=instance["N"]
    D=instance["D"]
    return len(N) * len(D) * np.log(2)

def CFL_MILP_h(instance, r ):
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
    w2 = model.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, name="w2")

    # Objective function
    obj = gp.quicksum(q[n] * (gp.quicksum(w[n,d] for d in D)) for n in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    
    model.addConstrs((w2[n]+ gp.quicksum(w[n,d] for d in D)<=1 for n in N))

    model.addConstrs(
            a[n,d]*(w[n, d]-z[d])+b[n]*w[n,d] <= 0
            for n in N for d in D)
    
    model.addConstrs((w[n, d] - a[n,d]/b[n]*w2[n]<= 0 for n in N for d in D), name="w_upper_bound_1")
    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, "Budget")


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
        #print("w:", {(n, d): w[n, d].x for n in N for d in D if w[n, d].x > 1e-6})
    else:
        print("No optimal solution found.")
        z_sol=[]
        w_sol=[]
        captured_demand=[]
    return z_sol,w_sol, captured_demand
    


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
    
    
    # Solve
    #model.setParam('FeasibilityTol', 1e-7)
    model.setParam('TimeLimit', 3600)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        z_sol_new={d: z[d].x for d in D}
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        Wdist=model.ObjVal
    elif model.status == gp.GRB.TIME_LIMIT:
        print("No optimal solution found.")
        z_sol_new={d: z[d].x for d in D}

        
        Prob_sol={ N[i]: {
        **{D[d]: Pd[N[i],D[d]].x  for d in range(len(D))},
        **{E[e]: Pe[N[i],E[e]].x for e in range(len(E))}
            } for i in range(len(N))}
        captured_demand=sum(q[n]* sum(Pd[n,d].x for d in D) for n in N)
        Wdist=model.ObjVal
    else:
        z_sol_new=[]
        Prob_sol=[]
        captured_demand=[]
        Wdist=0
    
    return z_sol_new, Prob_sol, captured_demand, Wdist



def feasible_solution_facility(instance, r, z_sol, alpha, dspace, captured_demand,distF,tipo):
    #looking for a feasible solution

    # ------------------------------
    # Step 1: Fix z and select top facilities
    # ------------------------------
    # Select facilities capturing the most demand
    instance_copy=copy.deepcopy(instance)
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    a0=instance["a"]
    b=instance["b"]
    be=instance["be"]
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    

    z={d: 0 for d in D}

    for d in dspace.keys():
        z[d]=1


    captured_d = {d: sum(q[n] * a0[n, d] * z_sol[d] / (sum(a0[n, d] * z_sol[d] for d in D) + b[n]) for n in N) for d in D}

    
    # Select top facilities
    top_indices = sorted(captured_d, key=captured_d.get,reverse=True)[0:r-1]
    
    for d in top_indices:
        z[d]=1

    
    # ------------------------------
    # Step 2: Calculate \bar a
    # ------------------------------
    def new_captured(constant,tipo):
        if tipo=="I":
            a_bar={d: 1 if d not in dspace.keys() else constant for n in N for d in D}
        elif tipo=="II":
            a_bar={d: 1 if d not in dspace.keys() and d not in top_indices else constant for n in N for d in D}
        return sum( sum(a_bar[ d] * a0[n, d] * z[d] for d in D) / (sum(a_bar[ d] * a0[n, d] * z[d] for d in D) + b[n]) for n in N)
    
    
    c=1
    while True:
        if new_captured(c,tipo)>= alpha*captured_demand:
            print('captured demand reached with c: '+str(c))
            c_sol=c
            break
        else:
            c+=0.01


    
    if tipo=="I":
        new_a={d: 1 if d not in dspace.keys()else c_sol for d in D}
        aux_new_a={(n,d):a0[n,d] if d not in dspace.keys()else c_sol*a0[n,d] for n in N for d in D}
    elif tipo=="II":
        new_a={d: 1 if d not in dspace.keys() and d not in top_indices else c_sol for d in D}
        aux_new_a={(n,d):a0[n,d] if d not in dspace.keys() and d not in top_indices else c_sol*a0[n,d] for n in N for d in D}
            
    # ------------------------------
    # Step 3: Wasserstein Distance Calculation
    # ------------------------------
    instance2=copy.deepcopy(instance_copy)
    instance2["a"]=aux_new_a
    P_new=get_probability_a(instance2,z)
    Wupper,pi_values= WassersteinDist(instance,get_probability_a(instance, z_sol),P_new,distF)


    # ------------------------------
    # Step 4: Auxiliary variables calculation: w_nd and u_nde
    # ------------------------------
    
    w_new={(n,d): aux_new_a[n,d]*z[d]/(sum(aux_new_a[n,k]*z[k]for k in D)+b[n]) for n in N for d in D}
    u_new={(n,e): be[n,e]/(sum(aux_new_a[n,k]*z[k] for k in D)+b[n])for n in N for e in E}

    # ------------------------------
    # Output results
    # ------------------------------
    initial_sol={}
    initial_sol['z_feas']=z
    initial_sol['a_feas']=new_a
    initial_sol['Wupper']=Wupper
    initial_sol['pi_feas']=pi_values
    initial_sol['w_feas']=w_new
    initial_sol['u_feas']=u_new

    return initial_sol



def retrieve_feature(instance, a_sol,beta_d, type):
    if type=="customer":
        feature = {
    (n, d): np.log(a_sol[(n, d)]) + beta_d * instance["theta_d"][i, j]
    for i, n in enumerate(instance["N"])
    for j, d in enumerate(instance["D"])
}
    elif type=="facility":
        feature = {
    d: np.log(a_sol[d]) for d in instance["D"]
}

    return feature




def RelativeExplanationsMixedFacility(instance,r,z_sol,alpha,dspace,Wbound,initial_sol,lams):
    
    # Parameters
    D = instance["D"]
    E = instance["E"]
    N = instance["N"]
    C=D+E
    q = {N[i]: instance["q"][i] for i in range(len(N))}
    a0=instance["a"]
    b=instance["b"]
    be=instance["be"]
    laml1=lams['L1']
    lamW=lams["W"]
   
    P0 = get_probability_a(instance, z_sol)
    
    # Model
    model = gp.Model("RelExp")
    
    # Variables
    a = model.addVars(D, vtype=GRB.CONTINUOUS, lb=1, name="a")
    aux_a = model.addVars( D, vtype=GRB.CONTINUOUS, name="a", lb=0)
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    w = model.addVars(N, D, vtype=GRB.CONTINUOUS,lb=0, name="w")
    #w2= model.addVars(N, vtype=GRB.CONTINUOUS,lb=0, name="w2")
    u = model.addVars (N,E, vtype=GRB.CONTINUOUS, name="u",lb=0,ub=1)
    pi = model.addVars(N, C, C, vtype=GRB.CONTINUOUS, name="pi")
    objvar=model.addVar(vtype=GRB.CONTINUOUS, lb=lamW*Wbound)

    #initial sol 
    if initial_sol!=[]:
        df = pd.DataFrame(list(initial_sol["u_feas"].items()), columns=['keys', 'value'])
        result = df.groupby(df['keys'].apply(lambda x: x[0]))['value'].sum().to_dict()
        for n in N:
            for d in D:
                a[d].start = initial_sol["a_feas"][d]
                w[n,d].start=initial_sol["w_feas"][(n,d)]
                z[d].start = initial_sol["z_feas"][d]
                for e in E:
                    u[n,e].start=initial_sol['u_feas'][(n,e)]
        for n in N:
            for c in C:
                for c_prime in C:
                    pi[n, c, c_prime].start = initial_sol['pi_feas'][(n, c, c_prime)]
    #Objective
    model.setObjective(objvar, GRB.MINIMIZE)
    
    # Constraints
    
    model.addConstr(objvar==laml1*gp.quicksum(aux_a[d] for d in D)+lamW/Wbound*gp.quicksum(pi[n, c, c_prime] * distFn(instance)[c,c_prime]
                                    for n in N for c in C for c_prime in C))

    model.addConstr(
        gp.quicksum(q[n] * (gp.quicksum(w[n,d] for d in D)) for n in N)
        >= alpha * gp.quicksum(q[n] * gp.quicksum(P0[n][d] for d in D) for n in N), name="captured_demand_control")
    
    model.addConstrs((gp.quicksum(pi[n, c, c_prime] for c_prime in C) == P0[n][c] 
                     for n in N for c in C), name="probability_constraints1")

   
    model.addConstrs((gp.quicksum(pi[n, c, d_prime] for c in C) == w[n,d_prime]
                      for n in N for d_prime in D), name="probability_constraints2")

    model.addConstrs((gp.quicksum(pi[n, c, e] for c in C) == u[n,e]
                      for n in N for e in E), name="probability_constraints3")
    
    model.addConstrs(
            gp.quicksum(w[n,d] for d in D)+gp.quicksum(u[n,e]for e in E)<=1
            for n in N)

    model.addConstrs (a0[n,d]*a[d]*(w[n,d]-z[d])+b[n]*w[n,d]<=0 
             for n in N for d in D)
    

    model.addConstrs((w[n,d]-a0[n,d]*a[d]/be[n,e]*u[n,e]<=0 for n in N for d in D for e in E))
    model.addConstrs((a0[n,d]*a[d]/be[n,e]*u[n,e]<=w[n,d]+(1-z[d]) for n in N for d in D for e in E))
    

    
    model.addConstr(gp.quicksum(z[d] for d in D) == r, name="z_cardinality")
    
    model.addConstrs(aux_a[d]>=(1-a[d]) for d in D)
    model.addConstrs(aux_a[d]>=-(1-a[d])  for d in D)

    # define D
    #model.addConstr(z['d1']==1, name="desired space")
    for key, value in dspace.items():
        z[key].LB = value
        z[key].UB = value
    
    # Spove
    #model.setParam('FeasibilityTol', 1e-9)
    timers = {
        "start_time": time.time(),
        "best_solution_time": None,
        "last_lower_bound_time": None,
          }
    def callback_with_data(model, where):
        my_callback(model, where, timers)


    model.setParam('TimeLimit', 3600)
    model.optimize(callback=callback_with_data)
    total_runtime = model.Runtime


    gap=model.MIPGap

    if model.status == gp.GRB.OPTIMAL or  model.SolCount > 0:
        z_sol_new={d: z[d].x for d in D}
        a_new={d: a[ d].X for d in D}
        w_sol={(n,d): w[n,d].X for n in N for d in D}
        captured_demand=sum(q[n] * sum(w_sol[n, d] for d in D) for n in N)
        objective=model.ObjVal
        total_time = model.Runtime
        gap=model.MIPGap
        

    else:
        z_sol_new=[]
        a_new=[]
        w_sol=[]
        captured_demand=[]
        objective=[]
        total_time = []
        gap=[]

    return z_sol_new, a_new, w_sol,captured_demand,objective, total_runtime,timers,gap





def my_callback(model, where, timers):
    current_time = time.time() - timers["start_time"]
    if where == gp.GRB.Callback.MIPSOL:
        objval = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        bound= model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)

        if "last_upper_bound" not in timers or objval < timers["last_upper_bound"]:
            timers["last_upper_bound"] = objval
            timers["best_solution_time"] = current_time
        if "last_lower_bound" not in timers or bound > timers["last_lower_bound"]:
            timers["last_lower_bound"] = bound
            timers["last_lower_bound_time"] = current_time

        elif where == gp.GRB.Callback.MIP:
            print("Entrando en MIP callback")
            bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

            # Guardar la mejor cota inferior
            if "last_lower_bound" not in timers or bound > timers["last_lower_bound"]:
                timers["last_lower_bound"] = bound
                timers["last_lower_bound_time"] = current_time
    
