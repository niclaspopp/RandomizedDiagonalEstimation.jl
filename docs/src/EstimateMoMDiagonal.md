## The EstimateMoMDiagonal Function

`EstimateMoMDiagonal` combines the functionality of `EstimateDiagonal` with the median of means principle.

```@docs
EstimateMoMDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, ngroups::Int, groupsize::Int, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int,O=nothing)
```
