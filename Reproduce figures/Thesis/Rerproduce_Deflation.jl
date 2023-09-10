using RandomizedPreconditioners
using DataFrames, StatsBase, Plots, Distributions, Statistics, LinearAlgebra, PGFPlotsX, LaTeXStrings, RandomizedLinAlg, Random, Measures
using RandomizedDiagonalEstimation
pgfplotsx()
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

diag_flat = diag(A_flat)
diag_poly = diag(A_poly)
diag_exp = diag(A_exp)
diag_step = diag(A_step)

norm_flat = norm(diag_flat)
norm_poly = norm(diag_poly)
norm_exp = norm(diag_exp)
norm_step = norm(diag_step);

## Test algorithms
n_tries = 50
range_test = 20:20:300
lim_q = length(range_test)

Gaussian_error_flat = zeros(lim_q)
Gaussian_error_poly = zeros(lim_q)
Gaussian_error_exp = zeros(lim_q)
Gaussian_error_step = zeros(lim_q)

Rademacher_error_flat = zeros(lim_q)
Rademacher_error_poly = zeros(lim_q)
Rademacher_error_exp = zeros(lim_q)
Rademacher_error_step = zeros(lim_q)

Shift_error_flat = zeros(lim_q)
Shift_error_poly = zeros(lim_q)
Shift_error_exp = zeros(lim_q)
Shift_error_step = zeros(lim_q)

NyDiagPP_error_flat = zeros(lim_q)
NyDiagPP_error_poly = zeros(lim_q)
NyDiagPP_error_exp = zeros(lim_q)
NyDiagPP_error_step = zeros(lim_q)

Diagpp_error_flat = zeros(lim_q)
Diagpp_error_poly = zeros(lim_q)
Diagpp_error_exp = zeros(lim_q)
Diagpp_error_step = zeros(lim_q)

XDiag_error_flat = zeros(lim_q)
XDiag_error_poly = zeros(lim_q)
XDiag_error_exp = zeros(lim_q)
XDiag_error_step = zeros(lim_q)

LR_error_flat = zeros(lim_q)
LR_error_poly = zeros(lim_q)
LR_error_exp = zeros(lim_q)
LR_error_step = zeros(lim_q)

for i=1:lim_q
    for j=1:n_tries
            Gaussian_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:GirardHutchinson, :queries, :Gaussian, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            Rademacher_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            Shift_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:GirardHutchinsonShift, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            NyDiagPP_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:NysDiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            Diagpp_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:DiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            XDiag_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:XDiag, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat
            LR_error_flat[i] += (1/n_tries)*norm(EstimateDiagonal(A_flat,:RSVD, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_flat)/norm_flat

            Gaussian_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:GirardHutchinson, :queries, :Gaussian, true,maxqueries=range_test[i])-diag_step)/norm_step
            Rademacher_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step
            Shift_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:GirardHutchinsonShift, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step
            NyDiagPP_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:NysDiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step
            Diagpp_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:DiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step
            XDiag_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:XDiag, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step
            LR_error_step[i] += (1/n_tries)*norm(EstimateDiagonal(A_step,:RSVD, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_step)/norm_step

            Gaussian_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:GirardHutchinson, :queries, :Gaussian, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            Rademacher_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            Shift_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:GirardHutchinsonShift, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            NyDiagPP_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:NysDiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            Diagpp_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:DiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            XDiag_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:XDiag, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly
            LR_error_poly[i] += (1/n_tries)*norm(EstimateDiagonal(A_poly,:RSVD, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_poly)/norm_poly

            Gaussian_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:GirardHutchinson, :queries, :Gaussian, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            Rademacher_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            Shift_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:GirardHutchinsonShift, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            NyDiagPP_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:NysDiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            Diagpp_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:DiagPP, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            XDiag_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:XDiag, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
            LR_error_exp[i] += (1/n_tries)*norm(EstimateDiagonal(A_exp,:RSVD, :queries, :Rademacher, true,maxqueries=range_test[i])-diag_exp)/norm_exp
    end
end

## Plot the results
l = @layout [a b;c d]

plot1 = scatter(range_test,Gaussian_error_flat,label="", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix vector products",color="red",yaxis=:log,legend=:topright,size=(1250,400),margin=10mm)
plot!(plot1,range_test,Gaussian_error_flat,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,Rademacher_error_flat,label="",color="green",yaxis=:log)
plot!(plot1,range_test,Rademacher_error_flat,label="",color="green",yaxis=:log)
scatter!(plot1,range_test,Shift_error_flat,label="",color="blue",yaxis=:log)
plot!(plot1,range_test,Shift_error_flat,label="",color="blue",yaxis=:log)
scatter!(plot1,range_test,Diagpp_error_flat,label="",color="yellow",yaxis=:log)
plot!(plot1,range_test,Diagpp_error_flat,label="",color="yellow",yaxis=:log)
scatter!(plot1,range_test,NyDiagPP_error_flat,label="",color="orange",yaxis=:log)
plot!(plot1,range_test,NyDiagPP_error_flat,label="",color="orange",yaxis=:log)
scatter!(plot1,range_test,XDiag_error_flat,label="",color="pink",yaxis=:log)
plot!(plot1,range_test,XDiag_error_flat,label="",color="pink",yaxis=:log)
scatter!(plot1,range_test,LR_error_flat,label="",color="brown",yaxis=:log)
plot!(plot1,range_test,LR_error_flat,label="",color="brown",yaxis=:log)

plot2 = scatter(range_test,Gaussian_error_step,label="GH (Rademacher)", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix vector products",yaxis=:log,color="red",legend=:topright,size=(1250,400),margin=10mm)
plot!(plot2,range_test,Gaussian_error_step,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,Rademacher_error_step,label="GH (Gaussian)",color="green",yaxis=:log)
plot!(plot2,range_test,Rademacher_error_step,label="",color="green",yaxis=:log)
scatter!(plot2,range_test,Shift_error_step,label="GH Shift",color="blue",yaxis=:log)
plot!(plot2,range_test,Shift_error_step,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,Diagpp_error_step,label="Diag++",color="yellow",yaxis=:log)
plot!(plot2,range_test,Diagpp_error_step,label="",color="yellow",yaxis=:log)
scatter!(plot2,range_test,NyDiagPP_error_step,label="NysDiag++",color="orange",yaxis=:log)
plot!(plot2,range_test,NyDiagPP_error_step,label="",color="orange",yaxis=:log)
scatter!(plot2,range_test,XDiag_error_step,label="XDiag",color="pink",yaxis=:log)
plot!(plot2,range_test,XDiag_error_step,label="",color="pink",yaxis=:log)
scatter!(plot2,range_test,LR_error_step,label="RSVD",color="brown",yaxis=:log)
plot!(plot2,range_test,LR_error_step,label="",color="brown",yaxis=:log)

plot3 = scatter(range_test,Gaussian_error_poly,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix vector products",yaxis=:log,color="red",legend=:topright,size=(1250,400),margin=10mm)
plot!(plot3,range_test,Gaussian_error_poly,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,Rademacher_error_poly,label="",color="green",yaxis=:log)
plot!(plot3,range_test,Rademacher_error_poly,label="",color="green",yaxis=:log)
scatter!(plot3,range_test,Shift_error_poly,label="",color="blue",yaxis=:log)
plot!(plot3,range_test,Shift_error_poly,label="",color="blue",yaxis=:log)
scatter!(plot3,range_test,Diagpp_error_poly,label="",color="yellow",yaxis=:log)
plot!(plot3,range_test,Diagpp_error_poly,label="",color="yellow",yaxis=:log)
scatter!(plot3,range_test,NyDiagPP_error_poly,label="",color="orange",yaxis=:log)
plot!(plot3,range_test,NyDiagPP_error_poly,label="",color="orange",yaxis=:log)
scatter!(plot3,range_test,XDiag_error_poly,label="",color="pink",yaxis=:log)
plot!(plot3,range_test,XDiag_error_poly,label="",color="pink",yaxis=:log)
scatter!(plot3,range_test,LR_error_poly,label="",color="brown",yaxis=:log)
plot!(plot3,range_test,LR_error_poly,label="",color="brown",yaxis=:log)


plot4 = scatter(range_test,Gaussian_error_exp,label="", title="Exp", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of matrix-vector products",yaxis=:log,color="red",legend=:topright,size=(1250,400),margin=10mm)
plot!(plot4,range_test,Gaussian_error_exp,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,Rademacher_error_exp,label="",color="green",yaxis=:log)
plot!(plot4,range_test,Rademacher_error_exp,label="",color="green",yaxis=:log)
scatter!(plot4,range_test,Shift_error_exp,label="",color="blue",yaxis=:log)
plot!(plot4,range_test,Shift_error_exp,label="",color="blue",yaxis=:log)
scatter!(plot4,range_test,Diagpp_error_exp,label="",color="yellow",yaxis=:log)
plot!(plot4,range_test,Diagpp_error_exp,label="",color="yellow",yaxis=:log)
scatter!(plot4,range_test,NyDiagPP_error_exp,label="",color="orange",yaxis=:log)
plot!(plot4,range_test,NyDiagPP_error_exp,label="",color="orange",yaxis=:log)
scatter!(plot4,range_test,XDiag_error_exp,label="",color="pink",yaxis=:log)
plot!(plot4,range_test,XDiag_error_exp,label="",color="pink",yaxis=:log)
scatter!(plot4,range_test,LR_error_exp,label="",color="brown",yaxis=:log)
plot!(plot4,range_test,LR_error_exp,label="",color="brown",yaxis=:log)

plot(plot1, plot2, plot3, plot4, layout = l,size=(1200,900))
