# RandomizedDiagonalEstimation.jl

## Overview

RandomizedDiagonalEstimation.jl is a Julia package that implements randomized methods for the estimation of the diagonal of matrices of matrices and matrix functions. For pure matrix diagonal estimation we provide the following algorithms
* Girard-Hutchinson Estimator
* Diag++
* NysDiag++
* XDiag
* Full Hutchinson Shifts
For the estimation of the diagonal of matrix function we combine the Girard-Hutchinson Estimator with the following approximations for ``f(\textbf(A))``
* Chebyshev interpolants to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Minimax polynomials from the Remez algorithm to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Arnoldi approximations

The package provides three functions: `EstimateDiagonal`, `EstimateFunctionDiagonal` and `EstimateMoMDiagonal`. The last function incorporates the median of means package into diagonal estimation.

## Estimating the Diagonal of Matrix

```@docs
EstimateDiagonal
```
