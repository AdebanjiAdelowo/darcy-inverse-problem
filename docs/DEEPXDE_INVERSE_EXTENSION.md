# DeepXDE inverse-PINN extension protocol

Status: PLANNED; no PINN inversion results claimed.

## Preserve the established baseline
Do not replace the FEM forward solver, verified adjoint, MAP experiments, or Bayesian pCN outputs. Existing Bayesian mixing and optimizer convergence caveats remain open.

## Proposed comparison
Use the same 2D Darcy equation, boundary conditions, log-permeability parameterization, and observation operator where mathematically compatible. Parameterize both pressure and log-permeability with neural networks and minimize PDE residual, observation misfit, boundary losses, and a specified regularizer. Compare against the existing FEM-adjoint MAP solution on identical synthetic observations, sensor placements, and noise levels.

## Scientific gates
Verify the PINN PDE residual sign and boundary flux; validate with a manufactured solution; prevent inverse-crime evaluation by using independent fine-mesh truth; record reconstruction L2 error, observation misfit, PDE residual, positivity, runtime, seeds and hyperparameter selection. A PINN comparison does not repair the Bayesian chain mixing issue.

Upstream: https://github.com/lululxvi/deepxde/tree/master/examples/pinn_inverse
