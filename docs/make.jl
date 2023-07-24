using Documenter
using RandomizedDiagonalEstimation

makedocs(
    sitename = "RandomizedDiagonalEstimation.jl",
    format = Documenter.HTML(),
    modules = [RandomizedDiagonalEstimation],
    pages = [
        "Home" => "index.md",
        "Estimation of the Diagonal of Matrices" => "EstimateDiagonal.md",
        "Median of Means Based Estimation" => "EstimateMoMDiagonal.md",
        "Estimation of the Diagonal of Matrix Functions" => "EstimateDiagonalFunction.md",
        "Examples" => "Examples.md"]
)

deploydocs(;
    repo="github.com/niclaspopp/RandomizedDiagonalEstimation.jl.git",
)
