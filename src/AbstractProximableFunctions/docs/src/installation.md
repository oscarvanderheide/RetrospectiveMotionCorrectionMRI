# Installation instructions

In the Julia REPL, simply type `]` and
```julia
(@v1.8) pkg> add https://github.com/grizzuti/AbstractLinearOperators.git, add https://github.com/grizzuti/AbstractProximableFunctions.git
```
Note that, since `AbstractProximableFunctions` has unregistered dependencies, we need to explicitly install also the package `AbstractLinearOperators`.