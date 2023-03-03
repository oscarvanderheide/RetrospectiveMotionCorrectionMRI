# Introduction

This package implements the algorithmic tools to perform retrospective motion correction for MRI.

``\mathrm{I}_{d\times d}`` 
```math
\hat{\nabla\mathbf{v}}|_i=\dfrac{\nabla\mathbf{v}|_i}{\sqrt{||\nabla\mathbf{v}|_i||_2^2+\eta^2}},
```

# Related publications

1. Ehrhardt, M. J., and Betcke, M. M., (2015). Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation (https://arxiv.org/abs/1511.06631), _SIAM J. IMAGING SCIENCES_, **9(3)**, 1084-1106, doi:[10.1137/15M1047325](https://doi.org/10.1137/15M1047325)
2. Rizzuti, G., Sbrizzi, A., and van Leeuwen, T., (2022). Joint Retrospective Motion Correction and Reconstruction for Brain MRI With a Reference Contrast, _IEEE Transaction on Computational Imaging_, **8**, 490-504, doi:[10.1109/TCI.2022.3183383](hhtps://doi.org/10.1109/TCI.2022.3183383)