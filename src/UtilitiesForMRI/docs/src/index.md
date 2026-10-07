# Introduction

This package implements some basic utilities for MRI, for example related to the definition of the:
- Cartesian ``x``-space discretization of a given field of view;
- ordered acquisition trajectory in ``k``-space;
- Fourier transform for several ``x``/``k``-space specifications.

Notably, `UtilitiesForMRI` allows the perturbation of the Fourier transform ``F`` with respect to some time-dependent rigid body motion ``\theta``. Furthermore, the mapping ``\theta\mapsto F(\theta)\mathbf{u}`` is differentiable, which can be useful for motion correction (see `[1]`).

## Relevant publications

1. Rizzuti, G., Sbrizzi, A., and van Leeuwen, T., (2022). Joint Retrospective Motion Correction and Reconstruction for Brain MRI With a Reference Contrast, _IEEE Transaction on Computational Imaging_, **8**, 490-504, doi:[10.1109/TCI.2022.3183383](hhtps://doi.org/10.1109/TCI.2022.3183383)