
# Relative Explanations for CFL 

This repository provides a modular framework for generating relative explanations of solutions to instances of the Competitive Facility Location (CFL) problem.

## 🧱 Structure
- `explanations_cfl/`: Core package containing generation of instances, utilities, and explanation tools.
- `experiments/`: Scripts to run experiments using the core package.

## 🚀 Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Running Experiments
```bash
python experiments/run_experiments.py
```

## 📦 Import Example
You can import all necessary functions directly from the package:
```python
from explanations_cfl import (
    generate_instance,utility_function, instance_coeff, get_preference_profiles,
    max_entropy_empirical, max_entropy_theorical, entropy_from_profiles,
    CFL_MILP_h, RelativeExplanationsMixedFacility, RelativeExplanationsDistGeneric_a,
    feasible_solution_facility, WassersteinDist, disFn,
    get_probability_a, retrieve_feature, my_callback
)
```

## 🛠 Notes
- Requires a valid [Gurobi](https://www.gurobi.com/) installation and license.
