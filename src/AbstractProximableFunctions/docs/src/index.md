# Introduction

This package provide some general abstract utilities for the computation of proximal and projection operators for the class of "proximable" functions (functions for which these operators can be implemented efficiently).

# Proximal and projection operators

The main functionalities provided by this package are related to the computation of the proximal and projection operators. For a generic (convex) functional ``g``, the proximal operator is defined as
```math
\mathrm{prox}_{\lambda,g}(\mathbf{w})=\arg\min_{\mathbf{u}}\dfrac{1}{2}||\mathbf{u}-\mathbf{w}||^2+\lambda{}g(\mathbf{u}).
```
Similarly, the projection operator is defined as
```math
\mathrm{proj}_{\varepsilon, g}(\mathbf{w})=\arg\min_{g(\mathbf{u})\le\varepsilon}\dfrac{1}{2}||\mathbf{u}-\mathbf{w}||^2.
```

# Why `AbstractProximableFunctions`?

Several properties of the proximal operators can be "abstractified", in a Julian sense, and are available once some basic components of the data type of interest are properly defined. As a trivial example, if a function `g::AbstractProximableFunction` implements a proximal operator method `prox(x, λ, g, ...)`, then the proximal operator of its convex conjugate ``ḡ`` (see `[1]`) is `prox(q, λ, ḡ, ...)=prox(q/λ, 1/λ, g, ...)`. A more interesting and ubiquitous example is when we want to combine a proximable function with a linear operator, e.g. ``g∘A``, which gives rise to a `WeightedProximableFunction` with some embedded default behavior.

# Related publications

1. Parikh, N., and Boyd, S., (2014). Proximal Algorithms, _Foundations and Trends in Optimization _, **1(3)**, 127-239, doi:[10.1561/2400000003](https://doi.org/10.1561/2400000003)