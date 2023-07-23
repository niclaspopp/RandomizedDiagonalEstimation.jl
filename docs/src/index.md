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

## Estimating the Diagonal of Matrix

The estimation of the diagonal of a function of a square matrix is handled by the function `EstimateDiagonal`

```@docs
EstimateFunctionDiagonal
```

## Estimating the Diagonal of Matrices Using Median of Means

`EstimateMoMDiagonal` combines the functionality of `EstimateDiagonal` with the median of means principle.

```@docs
EstimateMoMDiagonal
```

## Estimating the Diagonal of Matrix

The estimation of the diagonal of a function of a square matrix is handled by the function `EstimateDiagonal`

```@docs
EstimateFunctionDiagonal
```

## Citing RandomizedDiagonalEstimation.jl

If you use RandomizedDiagonalEstimation.jl for academic research and wish to cite it,
please cite the following paper (will be updated).

## References
TO DO
