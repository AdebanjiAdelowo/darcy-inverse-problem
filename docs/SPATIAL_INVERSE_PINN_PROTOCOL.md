# Spatial Darcy inverse PINN — manufactured parametric extension

Equation: `-div(k grad p)=f` on the unit square, zero Dirichlet boundary.
Manufactured pressure `p=sin(pi x) sin(pi y)`.
Positive spatial permeability `k=exp(alpha*x+beta*y)`; ground truth `alpha=0.5, beta=-0.3`.
The source `f=k*(2*pi²*p-alpha*p_x-beta*p_y)` is computed from the manufactured truth.

DeepXDE model jointly estimates pressure and the two permeability parameters using point observations and PDE residuals. The source depends only on ground-truth parameters, **not** inferred parameters.

This is a **two-parameter spatially varying coefficient benchmark**, not a high-dimensional KL random field and not an apples-to-apples replacement for the original FEM adjoint/Bayesian study. Full training must run remotely. Compare recovered alpha/beta, held-out pressure error, and permeability-field error; test noise and sensor counts in separate controlled experiments. Never claim successful recovery before verified runs.
