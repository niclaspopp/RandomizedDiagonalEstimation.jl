using DataFrames, StatsBase, Plots, Distributions, Statistics, LinearAlgebra, PGFPlotsX, LaTeXStrings, RandomizedLinAlg, Random, Measures
using ExponentialUtilities, Measures, RandomizedPreconditioners
using RandomizedDiagonalEstimation
pgfplotsx()
## include custom functions for these experiments
include("CustomFunctions.jl")
## Define Test Matrices
n=1000

flat_ev = zeros(n)
poly_ev = zeros(n)
exp_ev = zeros(n)
step_ev = zeros(n)

for i=1:n
    flat_ev[i] = 3-2(i-1)/(n-1)
    poly_ev[i] = i^(-2)
    exp_ev[i] = 0.7^(i)
    if i<=50
        step_ev[i]=1
    else
        step_ev[i]=10^(-3)
    end
end

V = rand(n,n)
Q, R = qr(V)

A_flat = Q*diagm(flat_ev)*Q'
A_poly = Q*diagm(poly_ev)*Q'
A_exp = Q*diagm(exp_ev)*Q'
A_step = Q*diagm(step_ev)*Q';























##
############################
# Exponential
############################
A_flat_exp  = Q*diagm(exp.(flat_ev))*Q'
A_poly_exp  = Q*diagm(exp.(poly_ev))*Q'
A_exp_exp  = Q*diagm(exp.(exp_ev))*Q'
A_step_exp  = Q*diagm(exp.(step_ev))*Q'

diag_flat_exp = diag(A_flat_exp)
diag_poly_exp = diag(A_poly_exp)
diag_exp_exp= diag(A_exp_exp)
diag_step_exp = diag(A_step_exp)

norm_flat_exp = norm(diag_flat_exp)
norm_poly_exp = norm(diag_poly_exp)
norm_exp_exp = norm(diag_exp_exp)
norm_step_exp = norm(diag_step_exp);

## Simple plot
fexp = x->exp(x)
deg = 25
range_test = 20:20:300
n_tries = 25
lim_q = length(range_test)

Chebyshev_error_flat_exp = zeros(lim_q)
Remez_error_flat_exp = zeros(lim_q)
Krylov_error_flat_exp = zeros(lim_q)

Chebyshev_approxerr_flat_exp = zeros(lim_q)
Remez_approxerr_flat_exp = zeros(lim_q)
Krylov_approxerr_flat_exp = zeros(lim_q)

Stocherr_flat_exp = zeros(lim_q)

Chebyshev_error_step_exp = zeros(lim_q)
Remez_error_step_exp = zeros(lim_q)
Krylov_error_step_exp = zeros(lim_q)

Chebyshev_approxerr_step_exp = zeros(lim_q)
Remez_approxerr_step_exp = zeros(lim_q)
Krylov_approxerr_step_exp = zeros(lim_q)

Stocherr_step_exp = zeros(lim_q)

Chebyshev_error_poly_exp = zeros(lim_q)
Remez_error_poly_exp = zeros(lim_q)
Krylov_error_poly_exp = zeros(lim_q)

Chebyshev_approxerr_poly_exp = zeros(lim_q)
Remez_approxerr_poly_exp = zeros(lim_q)
Krylov_approxerr_poly_exp = zeros(lim_q)

Stocherr_poly_exp = zeros(lim_q)

Chebyshev_approxerr_exp_exp = zeros(lim_q)
Remez_approxerr_exp_exp = zeros(lim_q)
Krylov_approxerr_exp_exp = zeros(lim_q)

Stocherr_exp_exp = zeros(lim_q)

Chebyshev_error_exp_exp = zeros(lim_q)
Remez_error_exp_exp = zeros(lim_q)
Krylov_error_exp_exp = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            ch_flat=EstimateFunctionDiagonal(A_flat,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_flat=EstimateFunctionDiagonal(A_flat,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_flat=EstimateFunctionDiagonal(A_flat,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, deg+1,maxqueries=range_test[i])
            GH_exact_exp_flat=EstimateDiagonal(A_flat_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_flat_exp[i] += (1/n_tries)*norm(ch_flat-diag_flat_exp)/norm_flat_exp
            Remez_error_flat_exp[i] += (1/n_tries)*norm(rem_flat-diag_flat_exp)/norm_flat_exp
            Krylov_error_flat_exp[i] += (1/n_tries)*norm(kr_flat-diag_flat_exp)/norm_flat_exp

            Stocherr_flat_exp[i] += (1/n_tries)*norm(GH_exact_exp_flat-diag_flat_exp)/norm_flat_exp

            Chebyshev_approxerr_flat_exp[i] += (1/n_tries)*norm(ch_flat-GH_exact_exp_flat)/norm_flat_exp
            Remez_approxerr_flat_exp[i] += (1/n_tries)*norm(rem_flat-GH_exact_exp_flat)/norm_flat_exp
            Krylov_approxerr_flat_exp[i] += (1/n_tries)*norm(kr_flat-GH_exact_exp_flat)/norm_flat_exp

            ch_step=EstimateFunctionDiagonal(A_step,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.001,1.0),maxqueries=range_test[i])
            rem_step=EstimateFunctionDiagonal(A_step,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.001,1.0),maxqueries=range_test[i])
            kr_step=EstimateFunctionDiagonal(A_step,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, deg+1,maxqueries=range_test[i])
            GH_exact_exp_step=EstimateDiagonal(A_step_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_step_exp[i] += (1/n_tries)*norm(ch_step-diag_step_exp)/norm_step_exp
            Remez_error_step_exp[i] += (1/n_tries)*norm(rem_step-diag_step_exp)/norm_step_exp
            Krylov_error_step_exp[i] += (1/n_tries)*norm(kr_step-diag_step_exp)/norm_step_exp

            Stocherr_step_exp[i] += (1/n_tries)*norm(GH_exact_exp_step-diag_step_exp)/norm_step_exp

            Chebyshev_approxerr_step_exp[i] += (1/n_tries)*norm(ch_step-GH_exact_exp_step)/norm_step_exp
            Remez_approxerr_step_exp[i] += (1/n_tries)*norm(rem_step-GH_exact_exp_step)/norm_step_exp
            Krylov_approxerr_step_exp[i] += (1/n_tries)*norm(kr_step-GH_exact_exp_step)/norm_step_exp

            ch_poly=EstimateFunctionDiagonal(A_poly,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.000001,1.0),maxqueries=range_test[i])
            rem_poly=EstimateFunctionDiagonal(A_poly,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.000001,1.0),maxqueries=range_test[i])
            kr_poly=EstimateFunctionDiagonal(A_poly,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, deg+1,maxqueries=range_test[i])
            GH_exact_exp_poly=EstimateDiagonal(A_poly_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_poly_exp[i] += (1/n_tries)*norm(ch_poly-diag_poly_exp)/norm_poly_exp
            Remez_error_poly_exp[i] += (1/n_tries)*norm(rem_poly-diag_poly_exp)/norm_poly_exp
            Krylov_error_poly_exp[i] += (1/n_tries)*norm(kr_poly-diag_poly_exp)/norm_poly_exp

            Stocherr_poly_exp[i] += (1/n_tries)*norm(GH_exact_exp_poly-diag_poly_exp)/norm_poly_exp

            Chebyshev_approxerr_poly_exp[i] += (1/n_tries)*norm(ch_poly-GH_exact_exp_poly)/norm_poly_exp
            Remez_approxerr_poly_exp[i] += (1/n_tries)*norm(rem_poly-GH_exact_exp_poly)/norm_poly_exp
            Krylov_approxerr_poly_exp[i] += (1/n_tries)*norm(kr_poly-GH_exact_exp_poly)/norm_poly_exp

            ch_exp=EstimateFunctionDiagonal(A_exp,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.0000000001,1.0),maxqueries=range_test[i])
            rem_exp=EstimateFunctionDiagonal(A_exp,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.0000000001,1.0),maxqueries=range_test[i])
            kr_exp=EstimateFunctionDiagonal(A_exp,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, deg+1,maxqueries=range_test[i])
            GH_exact_exp_exp=EstimateDiagonal(A_exp_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_exp_exp[i] += (1/n_tries)*norm(ch_exp-diag_exp_exp)/norm_exp_exp
            Remez_error_exp_exp[i] += (1/n_tries)*norm(rem_exp-diag_exp_exp)/norm_exp_exp
            Krylov_error_exp_exp[i] += (1/n_tries)*norm(kr_exp-diag_exp_exp)/norm_exp_exp

            Stocherr_exp_exp[i] += (1/n_tries)*norm(GH_exact_exp_exp-diag_exp_exp)/norm_exp_exp

            Chebyshev_approxerr_exp_exp[i] += (1/n_tries)*norm(ch_exp-GH_exact_exp_exp)/norm_exp_exp
            Remez_approxerr_exp_exp[i] += (1/n_tries)*norm(rem_exp-GH_exact_exp_exp)/norm_exp_exp
            Krylov_approxerr_exp_exp[i] += (1/n_tries)*norm(kr_exp-GH_exact_exp_exp)/norm_exp_exp

    end
end

##
plot1 = scatter(range_test,Chebyshev_error_flat_exp,markersize=5,label="", title="Flat spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix vector products",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot1,range_test,Chebyshev_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,Stocherr_flat_exp,label="",markersize=3,color="blue",yaxis=:log)
plot!(plot1,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot1,range_test,Chebyshev_approxerr_flat_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot1,range_test,Chebyshev_approxerr_flat_exp,label="",color="green",yaxis=:log)

plot2 = scatter(range_test,Remez_error_flat_exp,markersize=5,label="", title="Flat spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot2,range_test,Remez_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,Stocherr_flat_exp,markersize=3,label="",color="blue",yaxis=:log)
plot!(plot2,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,Remez_approxerr_flat_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot2,range_test,Remez_approxerr_flat_exp,label="",color="green",yaxis=:log)

plot3 = scatter(range_test,Krylov_error_flat_exp,markersize=5,label="Total Error", title="Flat spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot3,range_test,Krylov_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,Stocherr_flat_exp,markersize=3,label="Stochastic Error",color="blue",yaxis=:log )
plot!(plot3,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot3,range_test,Krylov_approxerr_flat_exp,markersize=3,label="Approximation Error",color="green",yaxis=:log)
plot!(plot3,range_test,Krylov_approxerr_flat_exp,label="",color="green",yaxis=:log)

plot4 = scatter(range_test,Chebyshev_error_step_exp,markersize=5,label="", title="Step spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot4,range_test,Chebyshev_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log)
plot!(plot4,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot4,range_test,Chebyshev_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot4,range_test,Chebyshev_approxerr_step_exp,label="",color="green",yaxis=:log)

plot5 = scatter(range_test,Remez_error_step_exp,markersize=5,label="", title="Step spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot5,range_test,Remez_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot5,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot5,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot5,range_test,Remez_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot5,range_test,Remez_approxerr_step_exp,label="",color="green",yaxis=:log)

plot6 = scatter(range_test,Chebyshev_error_step_exp,markersize=5,label="", title="Step spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.5),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot6,range_test,Chebyshev_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot6,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot6,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot6,range_test,Chebyshev_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot6,range_test,Chebyshev_approxerr_step_exp,label="",color="green",yaxis=:log)

plot7 = scatter(range_test,Chebyshev_error_poly_exp,markersize=5,label="", title="Poly spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot7,range_test,Chebyshev_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot7,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot7,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot7,range_test,Chebyshev_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot7,range_test,Chebyshev_approxerr_poly_exp,label="",color="green",yaxis=:log)

plot8 = scatter(range_test,Remez_error_poly_exp,markersize=5,label="", title="Poly spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot8,range_test,Remez_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot8,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot8,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot8,range_test,Remez_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot8,range_test,Remez_approxerr_poly_exp,label="",color="green",yaxis=:log)

plot9 = scatter(range_test,Krylov_error_poly_exp,markersize=5,label="", title="Poly spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot9,range_test,Krylov_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot9,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot9,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot9,range_test,Krylov_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot9,range_test,Krylov_approxerr_poly_exp,label="",color="green",yaxis=:log)

plot10 = scatter(range_test,Chebyshev_error_exp_exp,markersize=5,label="", title="Exp spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot10,range_test,Chebyshev_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot10,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot10,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot10,range_test,Chebyshev_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot10,range_test,Chebyshev_approxerr_exp_exp,label="",color="green",yaxis=:log)

plot11 = scatter(range_test,Remez_error_exp_exp,markersize=5,label="", title="Exp spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot11,range_test,Remez_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot11,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot11,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot11,range_test,Remez_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot11,range_test,Remez_approxerr_exp_exp,label="",color="green",yaxis=:log)

plot12 = scatter(range_test,Krylov_error_exp_exp,markersize=5,label="", title="Exp spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot12,range_test,Krylov_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot12,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot12,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot12,range_test,Krylov_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot12,range_test,Krylov_approxerr_exp_exp,label="",color="green",yaxis=:log)

l = @layout [a b c; d e f; g h i; j k l]
plot(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12,layout = l,size=(1000,1000))









##
############################
# Inverse
############################
A_flat_inv  = Q*diagm(inv.(flat_ev))*Q'
A_poly_inv  = Q*diagm(inv.(poly_ev))*Q'
A_exp_inv  = Q*diagm(inv.(exp_ev))*Q'
A_step_inv  = Q*diagm(inv.(step_ev))*Q'

diag_flat_inv = diag(A_flat_inv)
diag_poly_inv = diag(A_poly_inv)
diag_exp_inv= diag(A_exp_inv)
diag_step_inv = diag(A_step_inv)

norm_flat_inv = norm(diag_flat_inv)
norm_poly_inv = norm(diag_poly_inv)
norm_exp_inv = norm(diag_exp_inv)
norm_step_inv = norm(diag_step_inv);
## Simple plot
finv = x->inv(x)
deg = 25
range_test = 20:20:300
n_tries = 5
lim_q = length(range_test)

Chebyshev_error_flat_inv = zeros(lim_q)
Remez_error_flat_inv = zeros(lim_q)
Krylov_error_flat_inv = zeros(lim_q)

Chebyshev_approxerr_flat_inv = zeros(lim_q)
Remez_approxerr_flat_inv = zeros(lim_q)
Krylov_approxerr_flat_inv = zeros(lim_q)

Stocherr_flat_inv = zeros(lim_q)

Chebyshev_error_step_inv = zeros(lim_q)
Remez_error_step_inv = zeros(lim_q)
Krylov_error_step_inv = zeros(lim_q)

Chebyshev_approxerr_step_inv = zeros(lim_q)
Remez_approxerr_step_inv = zeros(lim_q)
Krylov_approxerr_step_inv = zeros(lim_q)

Stocherr_step_inv = zeros(lim_q)

Chebyshev_error_poly_inv = zeros(lim_q)
Remez_error_poly_inv = zeros(lim_q)
Krylov_error_poly_inv = zeros(lim_q)

Chebyshev_approxerr_poly_inv = zeros(lim_q)
Remez_approxerr_poly_inv = zeros(lim_q)
Krylov_approxerr_poly_inv = zeros(lim_q)

Stocherr_poly_inv = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            ch_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
            GH_exact_inv_flat=EstimateDiagonal(A_flat_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_flat_inv[i] += (1/n_tries)*norm(ch_flat-diag_flat_inv)/norm_flat_inv
            Remez_error_flat_inv[i] += (1/n_tries)*norm(rem_flat-diag_flat_inv)/norm_flat_inv
            Krylov_error_flat_inv[i] += (1/n_tries)*norm(kr_flat-diag_flat_inv)/norm_flat_inv

            Stocherr_flat_inv[i] += (1/n_tries)*norm(GH_exact_inv_flat-diag_flat_inv)/norm_flat_inv

            Chebyshev_approxerr_flat_inv[i] += (1/n_tries)*norm(ch_flat-GH_exact_inv_flat)/norm_flat_inv
            Remez_approxerr_flat_inv[i] += (1/n_tries)*norm(rem_flat-GH_exact_inv_flat)/norm_flat_inv
            Krylov_approxerr_flat_inv[i] += (1/n_tries)*norm(kr_flat-GH_exact_inv_flat)/norm_flat_inv

            ch_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.001,1.0),maxqueries=range_test[i])
            rem_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.001,1.0),maxqueries=range_test[i])
            kr_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
            GH_exact_inv_step=EstimateDiagonal(A_step_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_step_inv[i] += (1/n_tries)*norm(ch_step-diag_step_inv)/norm_step_inv
            Remez_error_step_inv[i] += (1/n_tries)*norm(rem_step-diag_step_inv)/norm_step_inv
            Krylov_error_step_inv[i] += (1/n_tries)*norm(kr_step-diag_step_inv)/norm_step_inv

            Stocherr_step_inv[i] += (1/n_tries)*norm(GH_exact_inv_step-diag_step_inv)/norm_step_inv

            Chebyshev_approxerr_step_inv[i] += (1/n_tries)*norm(ch_step-GH_exact_inv_step)/norm_step_inv
            Remez_approxerr_step_inv[i] += (1/n_tries)*norm(rem_step-GH_exact_inv_step)/norm_step_inv
            Krylov_approxerr_step_inv[i] += (1/n_tries)*norm(kr_step-GH_exact_inv_step)/norm_step_inv

            ch_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.000001,1.0),maxqueries=range_test[i])
            rem_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.000001,1.0),maxqueries=range_test[i])
            kr_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
            GH_exact_inv_poly=EstimateDiagonal(A_poly_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_poly_inv[i] += (1/n_tries)*norm(ch_poly-diag_poly_inv)/norm_poly_inv
            Remez_error_poly_inv[i] += (1/n_tries)*norm(rem_poly-diag_poly_inv)/norm_poly_inv
            Krylov_error_poly_inv[i] += (1/n_tries)*norm(kr_poly-diag_poly_inv)/norm_poly_inv

            Stocherr_poly_inv[i] += (1/n_tries)*norm(GH_exact_inv_poly-diag_poly_inv)/norm_poly_inv

            Chebyshev_approxerr_poly_inv[i] += (1/n_tries)*norm(ch_poly-GH_exact_inv_poly)/norm_poly_inv
            Remez_approxerr_poly_inv[i] += (1/n_tries)*norm(rem_poly-GH_exact_inv_poly)/norm_poly_inv
            Krylov_approxerr_poly_inv[i] += (1/n_tries)*norm(kr_poly-GH_exact_inv_poly)/norm_poly_inv
    end
end

##

plot1 = scatter(range_test,Chebyshev_error_flat_inv,markersize=5,label="", title="Flat spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot1,range_test,Chebyshev_error_flat_inv,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,Stocherr_flat_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot1,range_test,Stocherr_flat_inv,label="",color="blue",yaxis=:log)
scatter!(plot1,range_test,Chebyshev_approxerr_flat_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot1,range_test,Chebyshev_approxerr_flat_inv,label="",color="green",yaxis=:log)

plot2 = scatter(range_test,Remez_error_flat_inv,markersize=5,label="", title="Flat spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,14000.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot2,range_test,Remez_error_flat_inv,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,Stocherr_flat_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot2,range_test,Stocherr_flat_inv,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,Remez_approxerr_flat_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot2,range_test,Remez_approxerr_flat_inv,label="",color="green",yaxis=:log)

plot3 = scatter(range_test,Krylov_error_flat_inv,markersize=5,label="Total Error", title="Flat spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot3,range_test,Krylov_error_flat_inv,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,Stocherr_flat_inv,markersize=3,label="Stochastic Error",color="blue",yaxis=:log )
plot!(plot3,range_test,Stocherr_flat_inv,label="",color="blue",yaxis=:log)
scatter!(plot3,range_test,Krylov_approxerr_flat_inv,markersize=3,label="Approximation Error",color="green",yaxis=:log)
plot!(plot3,range_test,Krylov_approxerr_flat_inv,label="",color="green",yaxis=:log)

plot4 = scatter(range_test,Chebyshev_error_step_inv,markersize=5,label="", title="Step spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot4,range_test,Chebyshev_error_step_inv,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,Stocherr_step_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot4,range_test,Stocherr_step_inv,label="",color="blue",yaxis=:log)
scatter!(plot4,range_test,Chebyshev_approxerr_step_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot4,range_test,Chebyshev_approxerr_step_inv,label="",color="green",yaxis=:log)

plot5 = scatter(range_test,Remez_error_step_inv,markersize=5,label="", title="Step spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot5,range_test,Remez_error_step_inv,label="",color="red",yaxis=:log)
scatter!(plot5,range_test,Stocherr_step_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot5,range_test,Stocherr_step_inv,label="",color="blue",yaxis=:log)
scatter!(plot5,range_test,Remez_approxerr_step_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot5,range_test,Remez_approxerr_step_inv,label="",color="green",yaxis=:log)

plot6 = scatter(range_test,Krylov_error_step_inv,markersize=5,label="", title="Step spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.01,0.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot6,range_test,Krylov_error_step_inv,label="",color="red",yaxis=:log)
scatter!(plot6,range_test,Stocherr_step_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot6,range_test,Stocherr_step_inv,label="",color="blue",yaxis=:log)
scatter!(plot6,range_test,Krylov_approxerr_step_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot6,range_test,Krylov_approxerr_step_inv,label="",color="green",yaxis=:log)

plot7 = scatter(range_test,Chebyshev_error_poly_inv,markersize=5,label="", title="Poly spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot7,range_test,Chebyshev_error_poly_inv,label="",color="red",yaxis=:log)
scatter!(plot7,range_test,Stocherr_poly_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot7,range_test,Stocherr_poly_inv,label="",color="blue",yaxis=:log)
scatter!(plot7,range_test,Chebyshev_approxerr_poly_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot7,range_test,Chebyshev_approxerr_poly_inv,label="",color="green",yaxis=:log)

plot8 = scatter(range_test,Remez_error_poly_inv,markersize=5,label="", title="Poly spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot8,range_test,Remez_error_poly_inv,label="",color="red",yaxis=:log)
scatter!(plot8,range_test,Stocherr_poly_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot8,range_test,Stocherr_poly_inv,label="",color="blue",yaxis=:log)
scatter!(plot8,range_test,Remez_approxerr_poly_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot8,range_test,Remez_approxerr_poly_inv,label="",color="green",yaxis=:log)

plot9 = scatter(range_test,Krylov_error_poly_inv,markersize=5,label="", title="Poly spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot9,range_test,Krylov_error_poly_inv,label="",color="red",yaxis=:log)
scatter!(plot9,range_test,Stocherr_poly_inv,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot9,range_test,Stocherr_poly_inv,label="",color="blue",yaxis=:log)
scatter!(plot9,range_test,Krylov_approxerr_poly_inv,markersize=3,label="",color="green",yaxis=:log)
plot!(plot9,range_test,Krylov_approxerr_poly_inv,label="",color="green",yaxis=:log)

l = @layout [a b c; d e f; g h i]
plot(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9,layout = l,size=(1000,800))












##
############################
# Logarithm
############################
flog = x->log(x+1)
flogmat = A->log(A+Matrix(I,size(A)))
A_flat_log  = Q*diagm(flog.(flat_ev))*Q'
A_poly_log  = Q*diagm(flog.(poly_ev))*Q'
A_exp_log  = Q*diagm(flog.(exp_ev))*Q'
A_step_log  = Q*diagm(flog.(step_ev))*Q'

diag_flat_log = diag(A_flat_log)
diag_poly_log = diag(A_poly_log)
diag_exp_log = diag(A_exp_log)
diag_step_log = diag(A_step_log)

norm_flat_log = norm(diag_flat_log)
norm_poly_log = norm(diag_poly_log)
norm_exp_log = norm(diag_exp_log)
norm_step_log = norm(diag_step_log);
## Experiments
deg = 25
range_test = 20:20:300
n_tries = 1
lim_q = length(range_test)

Chebyshev_error_flat_exp = zeros(lim_q)
Remez_error_flat_exp = zeros(lim_q)
Krylov_error_flat_exp = zeros(lim_q)

Chebyshev_approxerr_flat_exp = zeros(lim_q)
Remez_approxerr_flat_exp = zeros(lim_q)
Krylov_approxerr_flat_exp = zeros(lim_q)

Stocherr_flat_exp = zeros(lim_q)

Chebyshev_error_step_exp = zeros(lim_q)
Remez_error_step_exp = zeros(lim_q)
Krylov_error_step_exp = zeros(lim_q)

Chebyshev_approxerr_step_exp = zeros(lim_q)
Remez_approxerr_step_exp = zeros(lim_q)
Krylov_approxerr_step_exp = zeros(lim_q)

Stocherr_step_exp = zeros(lim_q)

Chebyshev_error_poly_exp = zeros(lim_q)
Remez_error_poly_exp = zeros(lim_q)
Krylov_error_poly_exp = zeros(lim_q)

Chebyshev_approxerr_poly_exp = zeros(lim_q)
Remez_approxerr_poly_exp = zeros(lim_q)
Krylov_approxerr_poly_exp = zeros(lim_q)

Stocherr_poly_exp = zeros(lim_q)

Chebyshev_approxerr_exp_exp = zeros(lim_q)
Remez_approxerr_exp_exp = zeros(lim_q)
Krylov_approxerr_exp_exp = zeros(lim_q)

Stocherr_exp_exp = zeros(lim_q)

Chebyshev_error_exp_exp = zeros(lim_q)
Remez_error_exp_exp = zeros(lim_q)
Krylov_error_exp_exp = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            ch_flat=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_flat,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_flat=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_flat,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_flat=GirardHutchinsonFunctionDiagonalKrylovCustom(A_flat,range_test[i],flogmat,deg)
            GH_exact_exp_flat=RandomizedDiagonalEstimation.EstimateDiagonal(A_flat_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_flat_exp[i] += (1/n_tries)*norm(ch_flat-diag_flat_log)/norm_flat_log
            Remez_error_flat_exp[i] += (1/n_tries)*norm(rem_flat-diag_flat_log)/norm_flat_log
            Krylov_error_flat_exp[i] += (1/n_tries)*norm(kr_flat-diag_flat_log)/norm_flat_log

            Stocherr_flat_exp[i] += (1/n_tries)*norm(GH_exact_exp_flat-diag_flat_log)/norm_flat_log

            Chebyshev_approxerr_flat_exp[i] += (1/n_tries)*norm(ch_flat-GH_exact_exp_flat)/norm_flat_log
            Remez_approxerr_flat_exp[i] += (1/n_tries)*norm(rem_flat-GH_exact_exp_flat)/norm_flat_log
            Krylov_approxerr_flat_exp[i] += (1/n_tries)*norm(kr_flat-GH_exact_exp_flat)/norm_flat_log

            ch_step=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_step,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_step=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_step,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_step=GirardHutchinsonFunctionDiagonalKrylovCustom(A_step,range_test[i],flogmat,deg)
            GH_exact_exp_step=RandomizedDiagonalEstimation.EstimateDiagonal(A_step_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_step_exp[i] += (1/n_tries)*norm(ch_step-diag_step_log)/norm_step_log
            Remez_error_step_exp[i] += (1/n_tries)*norm(rem_step-diag_step_log)/norm_step_log
            Krylov_error_step_exp[i] += (1/n_tries)*norm(kr_step-diag_step_log)/norm_step_log

            Stocherr_step_exp[i] += (1/n_tries)*norm(GH_exact_exp_step-diag_step_log)/norm_step_log

            Chebyshev_approxerr_step_exp[i] += (1/n_tries)*norm(ch_step-GH_exact_exp_step)/norm_step_log
            Remez_approxerr_step_exp[i] += (1/n_tries)*norm(rem_step-GH_exact_exp_step)/norm_step_log
            Krylov_approxerr_step_exp[i] += (1/n_tries)*norm(kr_step-GH_exact_exp_step)/norm_step_log

            ch_poly=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_poly,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_poly=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_poly,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_poly=GirardHutchinsonFunctionDiagonalKrylovCustom(A_poly,range_test[i],flogmat,deg)
            GH_exact_exp_poly=RandomizedDiagonalEstimation.EstimateDiagonal(A_poly_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_poly_exp[i] += (1/n_tries)*norm(ch_poly-diag_poly_log)/norm_poly_log
            Remez_error_poly_exp[i] += (1/n_tries)*norm(rem_poly-diag_poly_log)/norm_poly_log
            Krylov_error_poly_exp[i] += (1/n_tries)*norm(kr_poly-diag_poly_log)/norm_poly_log

            Stocherr_poly_exp[i] += (1/n_tries)*norm(GH_exact_exp_poly-diag_poly_log)/norm_poly_log

            Chebyshev_approxerr_poly_exp[i] += (1/n_tries)*norm(ch_poly-GH_exact_exp_poly)/norm_poly_log
            Remez_approxerr_poly_exp[i] += (1/n_tries)*norm(rem_poly-GH_exact_exp_poly)/norm_poly_log
            Krylov_approxerr_poly_exp[i] += (1/n_tries)*norm(kr_poly-GH_exact_exp_poly)/norm_poly_log

            ch_exp=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_exp,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            rem_exp=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_exp,flogmat,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_exp=GirardHutchinsonFunctionDiagonalKrylovCustom(A_exp,range_test[i],flogmat,deg)
            GH_exact_exp_exp=RandomizedDiagonalEstimation.EstimateDiagonal(A_exp_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_exp_exp[i] += (1/n_tries)*norm(ch_exp-diag_exp_log)/norm_exp_log
            Remez_error_exp_exp[i] += (1/n_tries)*norm(rem_exp-diag_exp_log)/norm_exp_log
            Krylov_error_exp_exp[i] += (1/n_tries)*norm(kr_exp-diag_exp_log)/norm_exp_log

            Stocherr_exp_exp[i] += (1/n_tries)*norm(GH_exact_exp_exp-diag_exp_log)/norm_exp_log

            Chebyshev_approxerr_exp_exp[i] += (1/n_tries)*norm(ch_exp-GH_exact_exp_exp)/norm_exp_log
            Remez_approxerr_exp_exp[i] += (1/n_tries)*norm(rem_exp-GH_exact_exp_exp)/norm_exp_log
            Krylov_approxerr_exp_exp[i] += (1/n_tries)*norm(kr_exp-GH_exact_exp_exp)/norm_exp_log

    end
end

##
#plot(1:n,1:n)
plot1 = scatter(range_test,Chebyshev_error_flat_exp,markersize=5,label="", title="Flat spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix vector products",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot1,range_test,Chebyshev_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,Stocherr_flat_exp,label="",markersize=3,color="blue",yaxis=:log)
plot!(plot1,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot1,range_test,Chebyshev_approxerr_flat_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot1,range_test,Chebyshev_approxerr_flat_exp,label="",color="green",yaxis=:log)
# scatter!(plot1,range_test,Chebyshev_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log)
# plot!(plot1,range_test,Chebyshev_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot2 = scatter(range_test,Remez_error_flat_exp,markersize=5,label="", title="Flat spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,10000.),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot2,range_test,Remez_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,Stocherr_flat_exp,markersize=3,label="",color="blue",yaxis=:log)
plot!(plot2,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,Remez_approxerr_flat_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot2,range_test,Remez_approxerr_flat_exp,label="",color="green",yaxis=:log)
# scatter!(plot2,range_test,Remez_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log)
# plot!(plot2,range_test,Remez_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot3 = scatter(range_test,Krylov_error_flat_exp,markersize=5,label="Total Error", title="Flat spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.001,0.1),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot3,range_test,Krylov_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,Stocherr_flat_exp,markersize=3,label="Stochastic Error",color="blue",yaxis=:log )
plot!(plot3,range_test,Stocherr_flat_exp,label="",color="blue",yaxis=:log)
scatter!(plot3,range_test,Krylov_approxerr_flat_exp,markersize=3,label="Approximation Error",color="green",yaxis=:log)
plot!(plot3,range_test,Krylov_approxerr_flat_exp,label="",color="green",yaxis=:log)
# scatter!(plot3,range_test,Krylov_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log)
# plot!(plot3,range_test,Krylov_approxerr_flat_exp+Stocherr_flat_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot4 = scatter(range_test,Chebyshev_error_step_exp,markersize=5,label="", title="Step spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.1,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot4,range_test,Chebyshev_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log)
plot!(plot4,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot4,range_test,Chebyshev_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot4,range_test,Chebyshev_approxerr_step_exp,label="",color="green",yaxis=:log)
# scatter!(plot4,range_test,Chebyshev_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log)
# plot!(plot4,range_test,Chebyshev_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot5 = scatter(range_test,Remez_error_step_exp,markersize=5,label="", title="Step spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.1,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot5,range_test,Remez_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot5,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot5,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot5,range_test,Remez_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot5,range_test,Remez_approxerr_step_exp,label="",color="green",yaxis=:log)
# scatter!(plot5,range_test,Remez_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log)
# plot!(plot5,range_test,Remez_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot6 = scatter(range_test,Krylov_error_step_exp,markersize=5,label="", title="Step spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.1,2.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot6,range_test,Krylov_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot6,range_test,Stocherr_step_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot6,range_test,Stocherr_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot6,range_test,Krylov_approxerr_step_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot6,range_test,Krylov_approxerr_step_exp,label="",color="green",yaxis=:log)
# scatter!(plot6,range_test,Krylov_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log)
# plot!(plot6,range_test,Krylov_approxerr_step_exp+Stocherr_step_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot7 = scatter(range_test,Chebyshev_error_poly_exp,markersize=5,label="", title="Poly spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.5,10.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot7,range_test,Chebyshev_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot7,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot7,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot7,range_test,Chebyshev_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot7,range_test,Chebyshev_approxerr_poly_exp,label="",color="green",yaxis=:log)
# scatter!(plot7,range_test,Chebyshev_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log)
# plot!(plot7,range_test,Chebyshev_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot8 = scatter(range_test,Remez_error_poly_exp,markersize=5,label="", title="Poly spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.5,10.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot8,range_test,Remez_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot8,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot8,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot8,range_test,Remez_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot8,range_test,Remez_approxerr_poly_exp,label="",color="green",yaxis=:log)
# scatter!(plot8,range_test,Remez_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log)
# plot!(plot8,range_test,Remez_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot9 = scatter(range_test,Krylov_error_poly_exp,markersize=5,label="", title="Poly spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.5,10.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot9,range_test,Krylov_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot9,range_test,Stocherr_poly_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot9,range_test,Stocherr_poly_exp,label="",color="blue",yaxis=:log)
scatter!(plot9,range_test,Krylov_approxerr_poly_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot9,range_test,Krylov_approxerr_poly_exp,label="",color="green",yaxis=:log)
# scatter!(plot9,range_test,Krylov_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log)
# plot!(plot9,range_test,Krylov_approxerr_poly_exp+Stocherr_poly_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot10 = scatter(range_test,Chebyshev_error_exp_exp,markersize=5,label="", title="Exp spectrum, Chebyshev", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,1.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot10,range_test,Chebyshev_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot10,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot10,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot10,range_test,Chebyshev_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot10,range_test,Chebyshev_approxerr_exp_exp,label="",color="green",yaxis=:log)
# scatter!(plot10,range_test,Chebyshev_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log)
# plot!(plot10,range_test,Chebyshev_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot11 = scatter(range_test,Remez_error_exp_exp,markersize=5,label="", title="Exp spectrum, Remez", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,1.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot11,range_test,Remez_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot11,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot11,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot11,range_test,Remez_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot11,range_test,Remez_approxerr_exp_exp,label="",color="green",yaxis=:log)
# scatter!(plot11,range_test,Remez_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log)
# plot!(plot11,range_test,Remez_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log,linestyle=:dot)

plot12 = scatter(range_test,Krylov_error_exp_exp,markersize=5,label="", title="Exp spectrum, Krylov", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,ylims=(0.05,1.0),legend=:topright,size=(1250,400),margin=10pt)
plot!(plot12,range_test,Krylov_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot12,range_test,Stocherr_exp_exp,markersize=3,label="",color="blue",yaxis=:log )
plot!(plot12,range_test,Stocherr_exp_exp,label="",color="blue",yaxis=:log)
scatter!(plot12,range_test,Krylov_approxerr_exp_exp,markersize=3,label="",color="green",yaxis=:log)
plot!(plot12,range_test,Krylov_approxerr_exp_exp,label="",color="green",yaxis=:log)
# scatter!(plot12,range_test,Krylov_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log)
# plot!(plot12,range_test,Krylov_approxerr_exp_exp+Stocherr_exp_exp,label="",color="black",yaxis=:log,linestyle=:dot)


l = @layout [a b c; d e f; g h i; j k l]
plot(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12,layout = l,size=(1000,1000))
#savefig("log_three_errors_deg25.pdf")
