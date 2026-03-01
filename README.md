# Bayesian Optimization with Adaptive Trust Region

MATLAB implementation of modified Bayesian Optimization algorithm with adaptive trust region, selective pruning, and periodic exploration.

## Description

This code supports the undergraduate thesis:  
**"Black-box Optimization using Surrogate Models"**  
Author: Adheesha Pamudi Perera Mapitigama  
University of Colombo, Department of Mathematics (2026)

## Features

- Standard Bayesian Optimization with Gaussian Process surrogate
- Modified BO with:
  - Distance-based selective pruning
  - Adaptive trust region mechanism
  - Failure-triggered uncertainty exploration
- 7 benchmark test functions (Sphere, Rastrigin, Ackley, Branin, Beale, Goldstein-Price, Schaffer N.2)
- Comparative analysis tools

## Files

### Core Algorithms
- `bo_gp.m` - Standard Bayesian Optimization baseline
- `bo_gp_adaptiveTR_selective_pruning.m` - Modified algorithm

### Test Functions and Utilities
- `test_functions.m` - 7 benchmark optimization functions
- `run_comparison.m` - Single function comparison (2 algorithms)
- `error_cvg_all.m` - Error analysis across all functions
- `runtime_comparison_all_functions.m` - Runtime benchmarking
- `visualize_comparison.m` - Animated visualization tool

## Usage

### Quick Start
```matlab
% Load test functions
tests = test_functions();
test_case = tests(4);  % Branin function

% Run standard BO
out_std = bo_gp(test_case, 'EI', 100, 5, 1e-6, 0.1, 1e-12, 1000);

% Run modified algorithm
out_mod = bo_gp_adaptiveTR_selective_pruning(...
    test_case, 'EI', 100, 5, 1e-6, 0.1, 0.1, 0.9, 1e-12, 1000);

% Compare final errors
fprintf('Standard BO error: %.4e\n', abs(test_case.f_opt - out_std.y_best));
fprintf('Modified BO error: %.4e\n', abs(test_case.f_opt - out_mod.y_best));
```

### Run Full Comparison
```matlab
% Compare on specific function with visualization
run_comparison;

% Error analysis across all 7 functions (20 trials each)
error_cvg_all;

% Runtime comparison
runtime_comparison_all_functions;

% Animated visualization
visualize_comparison;
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 100 | Number of BO iterations |
| `n_initial` | 5 | Initial random samples |
| `sigma_n` | 1e-6 | GP noise variance |
| `ell_factor` | 0.1 | Length scale factor |
| `alpha_prune` | 0.1 | Pruning radius factor |
| `beta_trust` | 0.9 | Initial trust region factor |
| `epsilon` | 1e-12 | Convergence threshold |


## Results Summary

Average improvements over standard BO (20 trials, 100 iterations):

- **Error Reduction:** 45.8% average
  - Rastrigin: 45.1%
  - Schaffer N.2: 69.7%
  - Branin: 97.7%
  - Goldstein-Price: 79.3%
  
- **Runtime Reduction:** 11.4% average

## Citation

If you use this code, please cite:
```
@thesis{perera2026bayesian,
  author = {Adheesha Pamudi Perera Mapitigama},
  title = {Black-box Optimization using Surrogate Models},
  school = {University of Colombo},
  year = {2026},
  type = {Bachelor's Thesis}
}
```


## Contact

Adheesha Pamudi Perera Mapitigama  
Department of Mathematics, University of Colombo  
Email: [adheeperera@gmail.com]
