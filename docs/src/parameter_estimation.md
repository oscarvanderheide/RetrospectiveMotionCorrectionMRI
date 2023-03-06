# [Rigid motion parameter estimation](@id section-parest)

In this section, we lay out a detailed formulation of the optimization problem related to motion parameter estimation, given a *known* image. This is, essentially, a generalized rigid registration routine.

The mathematical problem considered is:
```math
\min_{\pmb{\theta}}J(\pmb{\theta})=\dfrac{1}{2}||F(\pmb{\theta})\mathbf{u}-\mathbf{d}||^2+\dfrac{\lambda^2}{2}||D\pmb{\theta}||^2.
```
The linear operator ``F(\pmb{\theta})`` is the motion-perturbed Fourier transform (described [here](@ref Fourier)), ``\mathbf{u}`` a fixed image, and ``\mathbf{d}`` some given data.

The regularization term is typically aimed at enforcing temporal smoothness. Its strength is regulated by the weight ``\lambda`` and regularization operator ``D`` (e.g. a time-derivative operator).

Temporal smoothness regularization can be integrated or replaced by hard constraints. It is convenient, for example, to define an time-wise interpolation operator ``I`` that transforms coarse-grid arrays ``\tilde{\pmb{\theta}}`` to fine-grid arrays ``\tilde{\pmb{\theta}}``. In this case, we solve the problem
```math
\min_{\tilde{\pmb{\theta}}}J(\tilde{\pmb{\theta}})=\dfrac{1}{2}||F(I\tilde{\pmb{\theta}})\mathbf{u}-\mathbf{d}||^2+\dfrac{\lambda^2}{2}||D\tilde{\pmb{\theta}}||^2.
```

## Solution method

We implemented a rather conventional pseudo-Hessian method for the optimization problem:
```math
\pmb{\theta}\leftarrow\pmb{\theta}-\mathrm{steplength}*B^{-1}\nabla_{\pmb{\theta}}J.
```
All the parameters described in this section can be set via the routine [`parameter_estimation_options`](@ref). Typically ``\mathrm{steplength}=1``.

The pseudo-Hessian ``B`` is based on the Gauss-Newton approximation of the analytical Hessian ``\nabla_{\pmb{\theta}}^2J``. Note, however that ``\nabla_{\pmb{\theta}}^2J`` is ill-conditioned for the wavenumber ``\mathbf{k}=0`` and high-frequency wavenumbers. Therefore, we use the Levenberg-Marquardt regularization:
```math
\tilde{B}=B+\mathrm{scaling}_{\mathrm{diag}}*\mathrm{diag}(B)+\mathrm{scaling}_{\mathrm{mean}}*\mathrm{mean}(\mathrm{diag}(B))\otimes\mathrm{Id}+\mathrm{scaling}_{\mathrm{id}}*\mathrm{Id}.
```
Here, the ``\mathrm{mean}`` is performed for the diagonals of each rigid-motion parameters.
Typically, ``\mathrm{scaling}_{\mathrm{id}}=0``, ``\mathrm{scaling}_{\mathrm{mean}}=\mathrm{1e-1}*\mathrm{scaling}_{\mathrm{diag}}`` and ``\mathrm{scaling}_{\mathrm{diag}}`` is a relatively small number.