## The EstimateDiagonal Function

The estimation of the diagonal of a function of a square matrix is handled by the function `EstimateDiagonal`

```@docs
EstimateDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int=0,O=nothing,var_est::Float64=1/eps(),epsilon::Float64=1.0, delta::Float64=1.0,con::Float64=1.0)
```
