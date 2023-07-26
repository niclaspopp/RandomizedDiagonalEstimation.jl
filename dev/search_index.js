var documenterSearchIndex = {"docs":
[{"location":"EstimateMoMDiagonal/#The-EstimateMoMDiagonal-Function","page":"Median of Means Based Estimation","title":"The EstimateMoMDiagonal Function","text":"","category":"section"},{"location":"EstimateMoMDiagonal/","page":"Median of Means Based Estimation","title":"Median of Means Based Estimation","text":"EstimateMoMDiagonal combines the functionality of EstimateDiagonal with the median of means principle.","category":"page"},{"location":"EstimateMoMDiagonal/","page":"Median of Means Based Estimation","title":"Median of Means Based Estimation","text":"EstimateMoMDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, ngroups::Int, groupsize::Int, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int,O=nothing)","category":"page"},{"location":"EstimateMoMDiagonal/#RandomizedDiagonalEstimation.EstimateMoMDiagonal","page":"Median of Means Based Estimation","title":"RandomizedDiagonalEstimation.EstimateMoMDiagonal","text":"function EstimateMoMDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, ngroups::Int, groupsize::Int, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int,O=nothing)\n\nFunction that computes a randomized estimate of the diagonal of a matrix using the median of means principle\n\nInput:\n\nA: Matrix to estimate the diagonal of\nAlgorithm: Choose which diagonal estimator is used, options are\n:GirardHutchinson\n:DiagPP\n:NysDiagPP\n:XDiag\n:XNysDiag\n:GirardHutchinsonShift\n:RSVD\n:AdaptiveDiagPP\nStoppingCriterion: How to terminate, possible options\nqueries: terminate when the maximum number of queries to A is reacher\nmaxqueries: maximum number of queries to A\nNote potential further stopping criteria will be added in future versions\ndistribution: Select the distribution from which the random vectors are drawn, inbuilt options are\n:Rademacher\n:Gaussion\n:custom, in this case provide a matrix with the test vectors as columns\nO: matrix with the test vectors as columns\nngroups: Number of groups for the Median of Means estimator\ngroupsize: Groupsize for the Median of Means estimator\n\n\n\n\n\n","category":"function"},{"location":"EstimateDiagonal/#The-EstimateDiagonal-Function","page":"Estimation of the Diagonal of Matrices","title":"The EstimateDiagonal Function","text":"","category":"section"},{"location":"EstimateDiagonal/","page":"Estimation of the Diagonal of Matrices","title":"Estimation of the Diagonal of Matrices","text":"The estimation of the diagonal of a function of a square matrix is handled by the function EstimateDiagonal.","category":"page"},{"location":"EstimateDiagonal/","page":"Estimation of the Diagonal of Matrices","title":"Estimation of the Diagonal of Matrices","text":"EstimateDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int=0,O=nothing,var_est::Float64=1/eps(),epsilon::Float64=1.0, delta::Float64=1.0,con::Float64=1.0)","category":"page"},{"location":"EstimateDiagonal/#RandomizedDiagonalEstimation.EstimateDiagonal","page":"Estimation of the Diagonal of Matrices","title":"RandomizedDiagonalEstimation.EstimateDiagonal","text":"function EstimateDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int=0,O=nothing,var_est::Float64=1/eps(),epsilon::Float64=1.0, delta::Float64=1.0,con::Float64=1.0)\n\nMain function to compute a randomized estimate of the diagonal of a matrix A\n\nInput:\n\nA: Matrix to estimate the diagonal of\nAlgorithm: Choose which diagonal estimator is used, options are\n:GirardHutchinson\nDiagPP\n:NysDiagPP\n:XDiag\n:GirardHutchinsonShift\n:RSVD\n:AdaptiveDiagPP\nStoppingCriterion: How to terminate, possible options\ndoubling: Use doubling strategy and terminate when the relative error estimate is below a threshold eps, required parameter\nqueries_start: number of queries to start with\neps: bound for the relative error estimate\nNote: has only been implemented with :GirardHutchinson and :DiagPP, other methods will be added with future releases\nqueries: terminate when the maximum number of queries to A is reacher\nmaxqueries: maximum number of queries to A\nadaptive: Using an epsilon delta estimator with :AdaptiveDiagPP\nepsdelta: parameters (epsilon,delta) for an epsilon-delta estimator\nmaxiter: maximum iterations\ndistribution: Select the distribution from which the random vectors are drawn, inbuilt options are\n:Rademacher\n:Gaussion\n:custom, in this case provide a matrix with the test vectors as columns\nO: matrix with the test vectors as columns\n\n\n\n\n\n","category":"function"},{"location":"EstimateDiagonalFunction/#The-EstimateFunctionDiagonal-Function","page":"Estimation of the Diagonal of Matrix Functions","title":"The EstimateFunctionDiagonal Function","text":"","category":"section"},{"location":"EstimateDiagonalFunction/","page":"Estimation of the Diagonal of Matrix Functions","title":"Estimation of the Diagonal of Matrix Functions","text":"The estimation of the diagonal of a function of a square matrix is handled by the function EstimateFunctionDiagonal.","category":"page"},{"location":"EstimateDiagonalFunction/","page":"Estimation of the Diagonal of Matrix Functions","title":"Estimation of the Diagonal of Matrix Functions","text":"EstimateFunctionDiagonal(A::Matrix{Float64},f,Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, MatFuncApprox::Symbol, deg::Int64, normalizationParam::Bool=true;maxqueries::Int,int::Tuple=(0.0,1.0),O=nothing)","category":"page"},{"location":"EstimateDiagonalFunction/#RandomizedDiagonalEstimation.EstimateFunctionDiagonal","page":"Estimation of the Diagonal of Matrix Functions","title":"RandomizedDiagonalEstimation.EstimateFunctionDiagonal","text":"function EstimateFunctionDiagonal(A::Matrix{Float64},f,Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, MatFuncApprox::Symbol, deg::Int64, normalizationParam::Bool=true;maxqueries::Int,int::Tuple=(0.0,1.0),O=nothing)\n\nMain function to compute a randomized estimate of the diagonal of a matrix function\n\nInput:\n\nA: Matrix as input for the function\nf: Function to approximate diagonal of\nAlgorithm: Choose which diagonal estimator is used, options are\n:GirardHutchinson\nMatFuncApprox: How to approximate f(A)b\nChebshev: Using Chebyshev polynomials\nrequires interval int and degree deg\nRemez: Using Remez polynomials\nrequires interval int and degree deg\nKrylov: Using the Arnoldi approximation\nrequires maximum potency of A in the Krylov subspace denoted by deg\nCG: Use conjugate gradient method to approximate diagonal of the inverse, Attention: f is neglected\nrequires maximum potency of A in the Krylov subspace denoted by deg\nStoppingCriterion: How to terminate, possible options\ndoubling: Use doubling strategy and terminate when the relative error estimate is below a threshold eps, required parameter\nqueries_start: number of queries to start with\neps: bound for the relative error estimate\nqueries: terminate when the maximum number of queries to A is reacher\nmaxqueries: maximum number of queries to A\ndistribution: Select the distribution from which the random vectors are drawn, inbuilt options are\n:Rademacher\n:Gaussion\n:custom, in this case provide a matrix with the test vectors as columns\nO: matrix with the test vectors as columns\ndeg: degree of polynomial or maximum potency of A in the Krylov subspace, depending on MatFuncApprox\n\n\n\n\n\n","category":"function"},{"location":"Examples/#Diagonal-Estimation","page":"Examples","title":"Diagonal Estimation","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The usage of the three function exported by the package is very simple. First, let's create an example matrix whose diagonal is easy to estimate.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"n=20\nOm = randn(n,n)\nQ,R = qr(Om)\nA_temp = Q*diagm(vec(1:n).^(-4))*Q'\nA_temp = A_temp*A_temp'\nA=A_temp\nA = 10^(-8)*A_temp\nfor i=1:n\n    A[i,i]=10^14*A[i,i]\nend\ntrue_diag = diag(A)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Standard diagonal estimation when using a fixed number of samples is carried out as follows.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Girard-Hutchinson\nEstimateDiagonal(A,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=100)≈true_diag\n# Diag++\nEstimateDiagonal(A,:DiagPP, :queries, :Rademacher, true,maxqueries=100)≈true_diag\n# NysDiag++\nEstimateDiagonal(A,:NysDiagPP, :queries, :Rademacher, true,maxqueries=100)≈true_diag\n# XDiag\nEstimateDiagonal(A,:XDiag, :queries, :Rademacher, true,maxqueries=100)≈true_diag","category":"page"},{"location":"Examples/#Doubling-strategy","page":"Examples","title":"Doubling strategy","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"When using the doubling strategy the third keyword has to be altered.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Girard-Hutchinson with doubling\nEstimateMoMDiagonal(A,:GirardHutchinson, :doubling, :Rademacher,true,queries_start=100,eps=0.2)\n# Diag++ with doubling\nEstimateMoMDiagonal(A,:DiagPP, :doubling, :Rademacher,true,queries_start=100,eps=0.2)","category":"page"},{"location":"Examples/#Median-of-Means","page":"Examples","title":"Median of Means","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Using the median of means versions with 100 subgroups of size 1000 is carried out by the EstimateMoMDiagonal function.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Girard-Hutchinson\nEstimateMoMDiagonal(A,:GirardHutchinson, :queries, :Rademacher, 100,900,true,maxqueries=100)\n# Diag++\nEstimateMoMDiagonal(A,:DiagPP, :queries, :Rademacher, 100,900,true,maxqueries=100)\n# NysDiag++\nEstimateMoMDiagonal(A,:NysDiagPP, :queries, :Rademacher, 100,900,true,maxqueries=100)\n# XDiag\nEstimateMoMDiagonal(A,:XDiag, :queries, :Rademacher, 100,900,true,maxqueries=100)","category":"page"},{"location":"Examples/#Adaptive-Diagonal-Estimation","page":"Examples","title":"Adaptive Diagonal Estimation","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"ADiag++ can be used by setting the second keyword to :DiagPP and the third keyword to :adaptive. The example shown next creates an (0.1,0.001) estimator.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Girard-Hutchinson\nRandomizedDiagonalEstimation.EstimateDiagonal(A,:DiagPP, :adaptive, :Rademacher, true,epsilon=0.1, delta=0.001)","category":"page"},{"location":"Examples/#Estimating-the-Diagonal-of-Matrix-Functions","page":"Examples","title":"Estimating the Diagonal of Matrix Functions","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"For testing the methods available matrix functions, we create another test matrix.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"fexp = x -> exp(x)\nOm = randn(n,n)\nQ,R = qr(Om)\nA_2 = Q*diagm(vec(1:n).^(-4))*Q'\nA_2_exp = Q*diagm(exp.(vec(1:n).^(-4)))*Q'\ndiag_exp = diag(A_2_exp)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"There are three different types of approximators for general matrix functions which can be used for the estimation of the diagonal in conjunction with the Girard-Hutchinson estimator. They all use the same syntax and are specified by the sixth keyword.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Remez Polynomials\nEstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, 25,int=(0.0,1.0),maxqueries=10000)\n# Chebyshev Interpolants\nEstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev,25,int=(0.0,1.0),maxqueries=10000)\n# Arnoldi Approximation\nEstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov,18,int=(0.0,1.0),maxqueries=10000)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"For the diagonal of the inverse we provide an additional estimator based on the conjugate gradients method.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"# Reference solution\nA_2_inv = Q*diagm((vec(1:n).^(2)))*Q'\ndiag_inv = diag(A_temp_2_inv)\n# Estimator\nfinv = x -> x^(-1)\nEstimateFunctionDiagonal(A_2,finv,:GirardHutchinson,:queries, :Gaussian, :CG, 25,int=(0.0,1.0),maxqueries=10000)","category":"page"},{"location":"#RandomizedDiagonalEstimation.jl","page":"Home","title":"RandomizedDiagonalEstimation.jl","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"RandomizedDiagonalEstimation.jl is a Julia package that implements randomized methods for the estimation of the diagonal of matrices and matrix functions. For pure matrix diagonal estimation the following algorithms are provided","category":"page"},{"location":"","page":"Home","title":"Home","text":"Girard-Hutchinson Estimator [1]\nDiag++ [2]\nNysDiag++\nXDiag [3]\nFull Hutchinson Shifts","category":"page"},{"location":"","page":"Home","title":"Home","text":"For the estimation of the diagonal of matrix function we combine the Girard-Hutchinson Estimator with the following approximations for f(textbfA)textbfb","category":"page"},{"location":"","page":"Home","title":"Home","text":"Chebyshev interpolants to approximate f on the interval lambda_minlambda_max\nMinimax polynomials from the Remez algorithm to approximate f on the interval lambda_minlambda_max\nArnoldi approximations","category":"page"},{"location":"","page":"Home","title":"Home","text":"The package exports three functions: EstimateDiagonal, EstimateFunctionDiagonal and EstimateMoMDiagonal. The last function incorporates the median of means package into diagonal estimation. A more detailed elaboration of the algorithms and theoretical properties can be found in this thesis: (will be updated once available on DIVA)","category":"page"},{"location":"#Citing-RandomizedDiagonalEstimation.jl","page":"Home","title":"Citing RandomizedDiagonalEstimation.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use RandomizedDiagonalEstimation.jl for academic research and wish to cite it, please cite the following paper (will be updated).","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"[1] C. Bekas, E. Kokiopoulou, and Y. Saad. “An estimator for the diagonal of a matrix”. In: Applied Numerical Mathematics 57.11 (2007). Numerical Algorithms, Parallelism and Applications (2), pp. 1214–1229. issn: 0168-9274.\n[2] R. A. Baston and Y. Nakatsukasa. “Stochastic diagonal estimation: probabilistic bounds and an improved algorithm”. In: ArXiv abs/2201.10684 (2022).\n[3] E. N. Epperly, J. A. Tropp, and R. J. Webber. “XTrace: Making the most of every sample in stochastic trace estimation”. In: arXiv e-prints, arXiv:2301.07825 (Jan. 2023), arXiv:2301.07825. arXiv: 2301. 07825 [math.NA].","category":"page"}]
}
