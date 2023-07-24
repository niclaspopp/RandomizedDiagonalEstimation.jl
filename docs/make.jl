using Documenter
using RandomizedDiagonalEstimation

makedocs(
    sitename = "RandomizedDiagonalEstimation",
    format = Documenter.HTML(),
    modules = [RandomizedDiagonalEstimation]
)

deploydocs(;
    repo="github.com/niclaspopp/RandomizedDiagonalEstimation.jl.git",
)
