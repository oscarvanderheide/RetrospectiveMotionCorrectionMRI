# FastSolversForWeightedTV

Set of utilities for computing proximal and projection operators with the TV regularization (and weighted version thereof, see reference). The methods are applicable to 1D, 2D and 3D problems and support GPU acceleration.

To install, run this command within the Julia REPL:
```
] add https://github.com/grizzuti/FastSolversForWeightedTV.git
```

See examples in the folder /examples for applications of total variation regularization via proximal or projection operators.

This package leverages the abstraction contained in [AbstractProximableFunctions.jl](https://github.com/grizzuti/AbstractProximableFunctions.git).


## References

 - Matthias J. Ehrhardt, Marta M. Betcke, "Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation", SIAM Journal on Imaging Sciences, 2016. https://arxiv.org/abs/1511.06631