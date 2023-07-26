## The EstimateFunctionDiagonal Function

The estimation of the diagonal of a function of a square matrix is handled by the function `EstimateFunctionDiagonal`.

```@docs
EstimateFunctionDiagonal(A::Matrix{Float64},fmat,f,Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, MatFuncApprox::Symbol, deg::Int64, normalizationParam::Bool=true;maxqueries::Int,int::Tuple=(0.0,1.0),O=nothing)
```
