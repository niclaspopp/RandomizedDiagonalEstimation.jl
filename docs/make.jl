using Documenter
using RandomizedDiagonalEstimation

makedocs(
    sitename = "RandomizedDiagonalEstimation",
    format = Documenter.HTML(),
    modules = [RandomizedDiagonalEstimation]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
