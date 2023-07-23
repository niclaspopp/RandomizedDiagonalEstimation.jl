# RandomizedDiagonalEstimation.jl

[![Build Status](https://travis-ci.com/niclaspopp/MultivariateDiscretization.jl.svg?branch=master)](https://travis-ci.com/niclaspopp/MultivariateDiscretization.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/niclaspopp/MultivariateDiscretization.jl?svg=true)](https://ci.appveyor.com/project/niclaspopp/MultivariateDiscretization-jl)
[![Coverage](https://codecov.io/gh/niclaspopp/MultivariateDiscretization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/niclaspopp/MultivariateDiscretization.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/niclaspopp/MultivariateDiscretization.jl/blob/master/MultivariateDiscretization/Doc%20v0.1.0.ipynb)
<br/>

## Overview

This is package a suite for the randomized estimation of matrix diagonals and matrix function diagonals written in Julia. The methods in the current version of  package
include:

- Girad-Hutchinson Diagonal Estimator
- Diag++
- NysDiag++
- XDiag
- ADiag++
- FunDiag with three different function approximators (Remez Polynomials, Chebyshevs interpolants and Arnoldi Approximations)

Algorithms which are currently in development and will be included in future versions of the package

- FunDiag++

## Documentation

For information on using the package,
[see the stable documentation](TO DO).

## Declaration, Contributing and Citing

The software in this packages was developed as part of my master thesis within the scope of only a few months. I have tested the code thoroughly to the best of my knowledge. However, if you find any bugs, I would be grateful if you let me know (e.g. through opening an issue) such that I can continuously improve the quality and stability of the package. Furthermore, I also intend to include future methods to keep the software as up-to-date as possible. In case you have developed a new method for randomized diagonal estimation, I would be very happy to receive a contribution and we can gladly include the method in the package. Otherwise, please feel free to star the repository to stay up to date. If you use RandomizedDiagonalEstimation.jl for research, teaching, or other activities, I would be grateful if you cite this work. [Link](TO DO).
