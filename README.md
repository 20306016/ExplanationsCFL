
# Relative Explanations for CFL 

This repository provides a framework for generating relative explanations of solutions to instances of the choice-based Competitive Facility Location (CFL) problem.

## Structure
- `explanations_cfl/`: Core package containing generation of instances, utilities, and the optimization models.
- `experiments/`: Scripts to run experiments using the core package.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶Running Experiments
```bash
python experiments/run_experiments.py
```

## Notes
- Requires a valid [Gurobi](https://www.gurobi.com/) installation and license.
