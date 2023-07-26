module RandomizedDiagonalEstimation

export EstimateDiagonal, EstimateFunctionDiagonal, EstimateMoMDiagonal

## load packages
using Random
using Statistics
using LinearAlgebra
using Distributions
using RandomizedLinAlg
using RandomizedPreconditioners
using Remez
using Polynomials
using Krylov
using ExponentialUtilities
using ApproxFun
using SpecialFunctions

## load files with sub-functions
include("FunctionDiagonalEstimation.jl")
include("PureDiagonalEstimation.jl")
include("ParallelizedEstimators.jl")
include("MoMEstimators.jl")
include("AdaptiveDiagonalEstimation.jl")
include("DoublingStrategy.jl")

# Define main functions
"""
    function EstimateDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int=0,O=nothing,var_est::Float64=1/eps(),epsilon::Float64=1.0, delta::Float64=1.0,con::Float64=1.0)

Main function to compute a randomized estimate of the diagonal of a matrix A

Input:
* A: Matrix to estimate the diagonal of
* Algorithm: Choose which diagonal estimator is used, options are
    * :GirardHutchinson
    * DiagPP
    * :NysDiagPP
    * :XDiag
    * :GirardHutchinsonShift
    * :RSVD
    * :AdaptiveDiagPP
* StoppingCriterion: How to terminate, possible options
    * doubling: Use doubling strategy and terminate when the relative error estimate is below a threshold eps, required parameter
        * queries_start: number of queries to start with
        * eps: bound for the relative error estimate
        * Note: has only been implemented with :GirardHutchinson and :DiagPP, other methods will be added with future releases
    * queries: terminate when the maximum number of queries to A is reacher
        * maxqueries: maximum number of queries to A
    * adaptive: Using an epsilon delta estimator with :AdaptiveDiagPP
        * epsdelta: parameters (epsilon,delta) for an epsilon-delta estimator
        * maxiter: maximum iterations
* distribution: Select the distribution from which the random vectors are drawn, inbuilt options are
    * :Rademacher
    * :Gaussion
    * :custom, in this case provide a matrix with the test vectors as columns
        * O: matrix with the test vectors as columns
"""
function EstimateDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int=0,O=nothing,var_est::Float64=1/eps(),epsilon::Float64=1.0, delta::Float64=1.0,con::Float64=1.0)

    # check if matrix is square
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    diag=zeros(n)
    # Check which stopping criterion is used
    if parallelizationParam==false
        if StoppingCriterion==:queries
            # Check if median of Means should be used
            if Algorithm==:GirardHutchinson
                diag = GirardHutchinsonDiagonal(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:DiagPP
                    diag = DiagPP(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:NysDiagPP
                diag = NysDiagPP(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:XDiag
                diag = XDiag(A,maxqueries,distribution,O)
                #diag = XDiag_old(A,maxqueries)
            elseif Algorithm==:XDiagEFF
                diag = XDiag_Efficient(A,maxqueries,distribution,O)
            elseif Algorithm==:GirardHutchinsonShift
                diag = GirardHutchinsonDiagonal_HutchinsonShift(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:RSVD
                diag = RSVD_Diag(A,maxqueries,distribution,O)
            else
                # Throw error: no suitable algorithm given
                ErrorException("No suitable algorithm selected")
            end

        elseif StoppingCriterion==:doubling
            if Algorithm==:GirardHutchinson
                diag,err_est,n_doublies = GirardHutchinsonDiagonal_Doubling(A,maxqueries,var_est,distribution,normalizationParam)
                print("Empirical variance: ",err_est)
            elseif Algorithm==:DiagPP
                     diag = DiagPP(A,maxqueries,distribution,normalizationParam,O)
            # elseif Algorithm==:NysDiagPP
            #     diag = NysDiagPP(A,maxqueries,distribution,normalizationParam,O)
            # elseif Algorithm==:XDiag
            #     diag = XDiag(A,maxqueries,distribution,O)
            #     #diag = XDiag_old(A,maxqueries)
            # elseif Algorithm==:XDiagEFF
            #     diag = XDiag_Efficient(A,maxqueries,distribution,O)
            # elseif Algorithm==:GirardHutchinsonShift
            #     diag = GirardHutchinsonDiagonal_HutchinsonShift(A,maxqueries,distribution,normalizationParam,O)
            # elseif Algorithm==:RSVD
            #     diag = RSVD_Diag(A,maxqueries,distribution,O)
            else
                # Throw error: no suitable algorithm given
                ErrorException("No suitable algorithm selected")
            end
        elseif StoppingCriterion==:adaptive
            diag = ADiagPP(A, epsilon, delta,con,maxqueries)
        else
            # Throw error: no suitable stopping criterion given
            ErrorException("No suitable stopping criterion given")
        end
    else
        if StoppingCriterion==:queries
            # Check if median of Means should be used
            if Algorithm==:GirardHutchinson
                diag = GirardHutchinsonDiagonal_Parallel(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:DiagPP
                    diag = DiagPP_Parallel(A,maxqueries,distribution,normalizationParam,O)
            elseif Algorithm==:XDiag
                diag = XDiag_Efficient_Parallel(A,maxqueries,distribution,O)
            elseif Algorithm==:GirardHutchinsonShift
                diag = GirardHutchinsonDiagonal_HutchinsonShift_Parallel(A,maxqueries,distribution,normalizationParam,O)
            else
                # Throw error: no suitable algorithm given
                ErrorException("No suitable algorithm selected")
            end
            # Note: NysDiagPP relies on other packages without much explicit computiation, thus not parallelized
            # Same goes for RSVD
        else
            # Throw error: no suitable stopping criterion given
            ErrorException("No suitable stopping criterion given")
        end
    end
    # Return final result
    return vec(diag)
end


"""
    function EstimateFunctionDiagonal(A::Matrix{Float64},fmat,f,Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, MatFuncApprox::Symbol, deg::Int64, normalizationParam::Bool=true;maxqueries::Int,int::Tuple=(0.0,1.0),q=4,O=nothing)

Main function to compute a randomized estimate of the diagonal of a matrix function

Input:
* A: Matrix as input for the function
* f: Function to approximate diagonal of
* Algorithm: Choose which diagonal estimator is used, options are
    * :GirardHutchinson
    * :funDiagPP
    * :funNys
* MatFuncApprox: How to approximate f(A)b
    * Chebshev: Using Chebyshev polynomials
        * requires interval int and degree deg
    * Remez: Using Remez polynomials
        * requires interval int and degree deg
    * Krylov: Using the Arnoldi approximation
        * requires maximum potency of A in the Krylov subspace denoted by deg
    * CG: Use conjugate gradient method to approximate diagonal of the inverse, Attention: f is neglected
        * requires maximum potency of A in the Krylov subspace denoted by deg
* StoppingCriterion: How to terminate, possible options
    * queries: terminate when the maximum number of queries to A is reacher
        * maxqueries: maximum number of queries to A
* distribution: Select the distribution from which the random vectors are drawn, inbuilt options are
    * :Rademacher
    * :Gaussion
    * :custom, in this case provide a matrix with the test vectors as columns
        * O: matrix with the test vectors as columns
* deg: degree of polynomial or maximum potency of A in the Krylov subspace, depending on MatFuncApprox
"""
function EstimateFunctionDiagonal(A::Matrix{Float64},fmat,f,Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, MatFuncApprox::Symbol, deg::Int64, normalizationParam::Bool=true;maxqueries::Int,int::Tuple=(0.0,1.0),q=4,O=nothing)

    # check if matrix is square
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    diag=zeros(n)
    # Check which stopping criterion is used
    if StoppingCriterion==:queries
        # Check if median of Means should be used
        if Algorithm==:GirardHutchinson
            if MatFuncApprox==:Chebyshev
                diag = GirardHutchinsonFunctionDiagonalPolyApprox(A,maxqueries,f,deg,int,distribution,normalizationParam,O)
            elseif MatFuncApprox==:Remez
                diag = GirardHutchinsonFunctionDiagonalRemez(A,maxqueries,f,deg,int,distribution,normalizationParam,O)
            elseif MatFuncApprox==:Krylov
                diag = GirardHutchinsonFunctionDiagonalKrylov(A,maxqueries,f,deg,distribution,normalizationParam,O)
            elseif MatFuncApprox==:CG
                diag = GirardHutchinsonDiagonalInverseCG(A,maxqueries,deg,distribution,normalizationParam,O)
            else
                ErrorException("No suitable stopping approximation algorithm for f(A)b selected")
            end
        elseif Algorithm==:funDiagPP
            if MatFuncApprox==:Chebyshev
                diag = funDiagPP_Chebyshev(A,fmat,f,maxqueries,deg,int,distribution,normalizationParam,q)
            elseif MatFuncApprox==:Remez
                diag = funDiagPP_Remez(A,fmat,f,maxqueries,deg,int,distribution,normalizationParam,q)
            elseif MatFuncApprox==:Krylov
                diag = funDiagPP_Krylov(A,fmat,f,maxqueries,deg,distribution,normalizationParam,q)
            else
                ErrorException("No suitable stopping approximation algorithm for f(A)b selected")
            end
        elseif Algorithm==:funNys
            diag = funNyströmDiag(A,f,maxqueries)
        else
            # Throw error: no suitable algorithm given
            ErrorException("No suitable algorithm selected")
        end
    elseif StoppingCriterion==:doubling
        ErrorException("Not implemented yet")
    else
        # Throw error: no suitable stopping criterion given
        ErrorException("No suitable stopping criterion given")
    end
    # Return final result
    return vec(diag)
end










"""
    function EstimateMoMDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, ngroups::Int, groupsize::Int, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int,O=nothing)

Function that computes a randomized estimate of the diagonal of a matrix using the median of means principle

Input:
* A: Matrix to estimate the diagonal of
* Algorithm: Choose which diagonal estimator is used, options are
    * :GirardHutchinson
    * :DiagPP
    * :NysDiagPP
    * :XDiag
    * :XNysDiag
    * :GirardHutchinsonShift
    * :RSVD
    * :AdaptiveDiagPP
* StoppingCriterion: How to terminate, possible options
    * queries: terminate when the maximum number of queries to A is reacher
        * maxqueries: maximum number of queries to A
    * Note potential further stopping criteria will be added in future versions
* distribution: Select the distribution from which the random vectors are drawn, inbuilt options are
    * :Rademacher
    * :Gaussion
    * :custom, in this case provide a matrix with the test vectors as columns
        * O: matrix with the test vectors as columns
* ngroups: Number of groups for the Median of Means estimator
* groupsize: Groupsize for the Median of Means estimator
"""
function EstimateMoMDiagonal(A::Matrix{Float64},Algorithm::Symbol, StoppingCriterion::Symbol, distribution::Symbol, ngroups::Int, groupsize::Int, normalizationParam::Bool=true, parallelizationParam::Bool=false ;maxqueries::Int,O=nothing)

    # check if matrix is square
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    diag=zeros(n)
    # Check which stopping criterion is used
    if parallelizationParam==false
        if StoppingCriterion==:queries
            # Check if median of Means should be used
            if Algorithm==:GirardHutchinson
                print(ngroups)
                diag = GirardHutchinsonDiagonalMoM(A,maxqueries,ngroups,groupsize,distribution,normalizationParam,O)
            elseif Algorithm==:DiagPP
                    diag = DiagPPMoM(A,maxqueries,ngroups,groupsize,distribution,normalizationParam,O)
            elseif Algorithm==:NysDiagPP
                diag = NysDiagPPMoM(A,maxqueries,ngroups,groupsize,distribution,normalizationParam,O)
            elseif Algorithm==:XDiag
                diag = XDiagMoM(A,maxqueries,ngroups,groupsize,distribution,O)
                #diag = XDiag_old(A,maxqueries)
            elseif Algorithm==:XDiagEFF
                diag = XDiag_EfficientMoM(A,maxqueries,ngroups,groupsize,distribution,O)
                #diag = XDiag_old(A,maxqueries)
            #elseif Algorithm==:XNysDiag
            #    diag = XNysDiagMoM(A,maxqueries,ngroups,groupsize,distribution,normalizationParam,O)
            elseif Algorithm==:GirardHutchinsonShift
                diag = GirardHutchinsonDiagonal_HutchinsonShiftMoM(A,maxqueries,ngroups,groupsize,distribution,normalizationParam,O)
            else
                # Throw error: no suitable algorithm given
                ErrorException("No suitable algorithm selected")
            end
        else
            # Throw error: no suitable stopping criterion given
            ErrorException("No suitable stopping criterion given")
        end
    else
        if StoppingCriterion==:queries
            # Check if median of Means should be used
            if Algorithm==:GirardHutchinson
                ErrorException("Not implemented yet")
            else
                # Throw error: no suitable algorithm given
                ErrorException("No suitable algorithm selected (or one that's not implemented yet)")
            end
        elseif StoppingCriterion==:doubling
            ErrorException("Not implemented yet")
        elseif StoppingCriterion==:adaptive
            ErrorException("Not implemented yet")
            #diag=AHutchPP(A, epsilon, delta,con,max_iter)
        else
            # Throw error: no suitable stopping criterion given
            ErrorException("No suitable stopping criterion given")
        end
    end
    # Return final result
    return vec(diag)
end


end
