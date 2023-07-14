# RandomizedDiagonalEstimation.jl

## Overview

RandomizedDiagonalEstimation.jl is a Julia package that implements randomized methods for the estimation of the diagonal of matrices of matrices and matrix functions. For pure matrix diagonal estimation we provide the following algorithms
* Girard-Hutchinson Estimator
* Diag++
* NysDiag++
* XDiag
* Full Hutchinson Shifts
For the estimation of the diagonal of matrix function we combine the Girard-Hutchinson Estimator with the following approximations for ``f(\textbf{A})\textbf{b}``
* Chebyshev interpolants to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Minimax polynomials from the Remez algorithm to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Arnoldi approximations

The package provides three functions: `EstimateDiagonal`, `EstimateFunctionDiagonal` and `EstimateMoMDiagonal`. The last function incorporates the median of means package into diagonal estimation. A more detailed elaboration of the algorithms and theoretical properties can be found in this thesis: (will be updated once available on DIVA)

## Estimating the Diagonal of Matrices

```@docs
EstimateDiagonal
```
## Estimating the Diagonal of Matrices Using Median of Means

```@docs
EstimateMoMDiagonal
```

## Estimating the Diagonal of Matrix
```@docs
EstimateFunctionDiagonal
```

## Examples
TO DO

## References
TO DO
