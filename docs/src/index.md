# RandomizedDiagonalEstimation.jl

## Overview

RandomizedDiagonalEstimation.jl is a Julia package that implements randomized methods for the estimation of the diagonal of matrices of matrices and matrix functions. For pure matrix diagonal estimation we provide the following algorithms
* Girard-Hutchinson Estimator [1]
* Diag++ [2]
* NysDiag++
* XDiag [3]
* Full Hutchinson Shifts
For the estimation of the diagonal of matrix function we combine the Girard-Hutchinson Estimator with the following approximations for ``f(\textbf{A})\textbf{b}``
* Chebyshev interpolants to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Minimax polynomials from the Remez algorithm to approximate ``f`` on the interval ``[\lambda_{min},\lambda_{max}]``
* Arnoldi approximations

The package provides three functions: `EstimateDiagonal`, `EstimateFunctionDiagonal` and `EstimateMoMDiagonal`. The last function incorporates the median of means package into diagonal estimation. A more detailed elaboration of the algorithms and theoretical properties can be found in this thesis: (will be updated once available on DIVA)


## Citing RandomizedDiagonalEstimation.jl

If you use RandomizedDiagonalEstimation.jl for academic research and wish to cite it,
please cite the following paper (will be updated).

## References
[1] C. Bekas, E. Kokiopoulou, and Y. Saad. “An estimator for the diagonal of a matrix”. In: Applied Numerical Mathematics 57.11 (2007). Numerical Algorithms, Parallelism and Applications (2), pp. 1214–1229. issn: 0168-9274.\
[2] R. A. Baston and Y. Nakatsukasa. “Stochastic diagonal estimation: probabilistic bounds and an
improved algorithm”. In: ArXiv abs/2201.10684 (2022).\
[3] E. N. Epperly and J. A. Tropp. Efficient error and variance estimation for randomized matrix com- putations. 2023. arXiv: 2207.06342 [math.NA].
