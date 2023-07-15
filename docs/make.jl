using Documenter
using RandomizedDiagonalEstimation

makedocs(
    sitename = "RandomizedDiagonalEstimation",
    format = Documenter.HTML(),
    modules = [RandomizedDiagonalEstimation],
    pages = [
        "Home" => "index.md",
        "Diagonal of Matrices" => "EstimateDiagonalDocs.md",
        "Median of Means" => "EstimateDiagonalMoMDocs.md",
        "Diagonal of Matrix Functions" => "EstimateDiagonalFunctionDocs.md",
        "Exampled" => "ExamplesDocs.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
