# [Motion-perturbed Fourier transform](@id Fourier)

In this section, we clarify the notion of motion-perturbed Fourier transform (more details can be found in the [reference](@ref intro) section `[1]`).

## The non-uniform Fourier transform

For a discretized (complex-valued) image ``\mathbf{u}`` on some computational grid ``\{\mathbf{x}_j\}_j``, that is
```math
\mathbf{u}(\mathbf{x})=\sum_{j=1}^n\mathbf{u}_j\delta(\mathbf{x}-\mathbf{x}_j),
```
the (non-uniform) Fourier transform is defined as:
```math
F\mathbf{u}\,(\mathbf{k})=\sum_j\mathbf{u}_j\exp{(2\pi\mathrm{i}\,\mathbf{k}\cdot\mathbf{x}_j)},\qquad\mathbf{k}\in K
```
for any wavenumber set ``K``. This can be evaluated by non-uniform Fourier libraries such as [`FINUFFT.jl`](https://github.com/ludvigak/FINUFFT.jl) or [`NFFT.jl`](https://github.com/JuliaMath/NFFT.jl).

## Global rigid motion under the Fourier transform

Whenever the image undergoes a rigid motion, that is ``\mathbf{u}_{T(\pmb{\theta})}=\mathbf{u}\circ T(\pmb{\theta})^{-1}`` where
```math
T(\pmb{\theta})(\mathbf{x})=R(\pmb{\varphi})\mathbf{x}+\pmb{\tau},\qquad\mathrm{where\ }\pmb{\theta}=(\pmb{\varphi},\pmb{\tau}),
```
we can define the *perturbed* Fourier transform of ``\mathbf{u}`` to be ``F(\pmb{\theta})\mathbf{u}:=F\mathbf{u}_{T(\pmb{\theta})}``. Thanks to elementary Fourier identities, this gives:
```math
F(\pmb{\theta})\mathbf{u}\,(\mathbf{k})=\exp{(-2\pi\mathrm{i}\,\mathbf{k}\cdot\pmb{\tau})}F\mathbf{u}\,(R(\pmb{\varphi})^{\mathrm{T}}\mathbf{k}).
```

## Time-dependent rigid motion under the Fourier transform

In MRI, the Fourier transform is evaluated in ``k``-space according to a certain temporal sequence of wavenumbers ``\{\mathbf{k}_t\}_{t=1}^{n_t}``. For motion correction purposes, we must then postulate (in principle) as many rigid motion parameters ``\pmb{\theta}_t`` as "time" steps, so with a slight abuse of notation we will consider as perturbed Fourier transform the following:
```math
F(\pmb{\theta})\mathbf{u}:=(F(\pmb{\theta}_1)\mathbf{u}(\mathbf{k}_1),\ldots,F(\pmb{\theta}_{n_t})\mathbf{u}(\mathbf{k}_{n_t})).
```
Note that, in practice, we can assume that ``\pmb{\theta}_t`` remains constant for time indexes associated to wavenumbers ``\mathbf{k}_t`` that belongs to the same readout. In general, we can also assume that the rigid motion parameters vary smoothly in time.