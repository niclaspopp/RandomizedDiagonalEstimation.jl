using ExponentialUtilities, MAT, BenchmarkTools, ApproxFun, Polynomials, Krylov, Remez
using DelimitedFiles, Random, Statistics, Roots, LinearAlgebra, Distributions, Statistics, Plots, ToeplitzMatrices, RandomizedLinAlg, Measures, LaTeXStrings
using RandomizedDiagonalEstimation
pgfplotsx()

## Import the matrix
A=matread("Applications/Estrada_Index/Barabasi_yeast.mat")["Problem"]["A"];
n=size(A)[1]

## Compute Matrix Exponential
Aexp = exp(Matrix(A))

diag_exp = diag(Aexp)
norm_exp=norm(diag_exp)
norm_inf=norm(diag_exp,Inf)
abs_diag_exp = abs.(diag_exp);

## Perform estimates
fexp = x->exp(x)
eig=eigen(Matrix(A))
rightbound=exp(maximum(eig.values))
##
n_tries = 3
range_test = 100:100:1000
lim_q = length(range_test)
degree = 10
fexp = x->exp(x)
##

# Exact_error = zeros(lim_q);
Expv_error = zeros(lim_q);
Chebyshev_error = zeros(lim_q);
Uniform_error = zeros(lim_q);
Krylov_error = zeros(lim_q);

# Exact_error_inf = zeros(lim_q);
Expv_error_inf = zeros(lim_q);
Chebyshev_error_inf = zeros(lim_q);
Uniform_error_inf = zeros(lim_q);
Krylov_error_inf = zeros(lim_q);

Chebyshev_error_trace = zeros(lim_q);
Uniform_error_trace = zeros(lim_q);
Krylov_error_trace = zeros(lim_q);

for i=1:lim_q
    for j=1:n_tries
        # queries_series = Int(floor(range_test[i]/degree))

        Krylov_res = EstimateFunctionDiagonal(Matrix(A),fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, degree,maxqueries=range_test[i])
        Chebyshev_res = EstimateFunctionDiagonal(Matrix(A),fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, degree,maxqueries=range_test[i])
        Uniform_res = EstimateFunctionDiagonal(Matrix(A),fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, degree+1,maxqueries=range_test[i])

        Chebyshev_error[i] +=  (norm(Chebyshev_res - vec(diag_exp))/norm_exp)/n_tries;
        Uniform_error[i] +=  (norm(Uniform_res - vec(diag_exp))/norm_exp)/n_tries;
        Krylov_error[i] +=  (norm(Krylov_res- vec(diag_exp))/norm_exp)/n_tries;

        Chebyshev_error_inf[i] +=  mean(abs.(Chebyshev_res - vec(diag_exp))./abs_diag_exp)/n_tries;
        Uniform_error_inf[i] +=  mean(abs.(Uniform_res - vec(diag_exp))./abs_diag_exp)/n_tries;
        Krylov_error_inf[i] +=  mean(abs.(Krylov_res - vec(diag_exp))./abs_diag_exp)/n_tries;

        Chebyshev_error_trace[i] +=  abs(sum(Chebyshev_res)-sum(vec(diag_exp)))/abs(sum(vec(diag_exp)))
        Uniform_error_trace[i] +=  abs(sum(Uniform_res)-sum(vec(diag_exp)))/abs(sum(vec(diag_exp)))
        Krylov_error_trace[i] +=  abs(sum(Krylov_res)-sum(vec(diag_exp)))/abs(sum(vec(diag_exp)))
    end
end

## Plot results
p1=plot(range_test,Uniform_error,xlabel="Number of test vectors",ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}",label="",yaxis=:log,color="red")
scatter!(p1,range_test,Uniform_error,label="",color="red")
plot!(p1,range_test,Krylov_error,label="",color="green")
scatter!(p1,range_test,Krylov_error,label="",color="green")
plot!(p1,range_test,Chebyshev_error,label="",color="blue")
scatter!(p1,range_test,Chebyshev_error,label="",color="blue")
p2=plot(range_test,Uniform_error_inf,xlabel="Number of test vectors",ylabel=L"\textrm{Mean relative }\mathcal{L}^{\infty}\textrm{ error}",label="Remez",yaxis=:log,color="red",legend=:topright)
scatter!(p2,range_test,Uniform_error_inf,label="",color="red")
plot!(p2,range_test,Chebyshev_error_inf,label="Chebyshev",color="blue")
scatter!(p2,range_test,Chebyshev_error_inf,label="",color="blue")
plot!(p2,range_test,Krylov_error_inf,label="Krylov",color="green")
scatter!(p2,range_test,Krylov_error_inf,label="",color="green")
l = @layout [a b]
plot(p1, p2, layout = l,size=(800,300))
