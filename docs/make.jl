using Documenter
using RandomizedDiagonalEstimation

makedocs(
    sitename = "RandomizedDiagonalEstimation",
    format = Documenter.HTML(),
    modules = [RandomizedDiagonalEstimation],
    pages = [
        "Home" => "index.md",
        "Examples" => "Examples.md"]
)

deploydocs(;
    repo="github.com/niclaspopp/RandomizedDiagonalEstimation.jl.git",
)
