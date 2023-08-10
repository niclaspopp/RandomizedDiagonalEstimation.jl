using RandomizedPreconditioners
using DataFrames, StatsBase, Plots, Distributions, Statistics, LinearAlgebra, PGFPlotsX, LaTeXStrings, RandomizedLinAlg, Random, Measures
using ExponentialUtilities, Measures, ApproxFun, SpecialFunctions, Polynomials, KrylovKit, Remez
using RandomizedDiagonalEstimation
pgfplotsx()
## include custom functions
include("./CustomFunctions.jl")

## Define Test Matrices
n=1000

flat_ev = zeros(n)
poly_ev = zeros(n)
exp_ev = zeros(n)
step_ev = zeros(n)

for i=1:n
    flat_ev[i] = 3-2(i-1)/(n-1)
    poly_ev[i] = i^(-3)
    exp_ev[i] = 0.5^(i)
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

## define normalized exponential functions
fexp = x->exp(x)
fexp_norm = x->exp(x).-1
fexp_norm_mat = x->exp(x).-Matrix(I,size(x))

## Simple plot
fexp = x->exp(x)
deg = 25
range_test = 20:20:300
n_tries = 8
lim_q = length(range_test)

# Chebyshev_error_flat_exp = zeros(lim_q)
# Remez_error_flat_exp = zeros(lim_q)
Krylov_error_flat_exp = zeros(lim_q)
FN_flat_exp = zeros(lim_q)
PP_kr_flat_exp = zeros(lim_q)
PP_ch_flat_exp = zeros(lim_q)
PP_rem_flat_exp = zeros(lim_q)
#
# Chebyshev_error_step_exp = zeros(lim_q)
# Remez_error_step_exp = zeros(lim_q)
Krylov_error_step_exp = zeros(lim_q)
FN_step_exp = zeros(lim_q)
PP_kr_step_exp = zeros(lim_q)
PP_ch_step_exp = zeros(lim_q)
PP_rem_step_exp = zeros(lim_q)
#
# Chebyshev_error_poly_exp = zeros(lim_q)
# Remez_error_poly_exp = zeros(lim_q)
Krylov_error_poly_exp = zeros(lim_q)
FN_poly_exp = zeros(lim_q)
PP_kr_poly_exp = zeros(lim_q)
PP_ch_poly_exp = zeros(lim_q)
PP_rem_poly_exp = zeros(lim_q)
# Chebyshev_error_exp_exp = zeros(lim_q)
# Remez_error_exp_exp = zeros(lim_q)
Krylov_error_exp_exp = zeros(lim_q)
FN_exp_exp = zeros(lim_q)
PP_kr_exp_exp = zeros(lim_q)
PP_ch_exp_exp = zeros(lim_q)
PP_rem_exp_exp = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            # ch_flat=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_flat,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
            # rem_flat=RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_flat,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_flat=GirardHutchinsonFunctionDiagonalExpNorm(A_flat,range_test[i],deg).+1
            FN_flat=funNyströmDiag(A_flat,fexp_norm,range_test[i]).+1
            PP_kr_flat=funDiagPPExp(A_flat,fexp_norm,range_test[i],deg).+1
            int_flat=(1.0,3.0)
            PP_ch_flat=ChebyshevfunDiagPP(A_flat,fexp_norm_mat,fexp_norm,range_test[i],deg,int_flat).+1
            PP_rem_flat=RemezfunDiagPP(A_flat,fexp_norm_mat,fexp_norm,range_test[i],deg,int_flat).+1

            Krylov_error_flat_exp[i] += (1/n_tries)*norm(kr_flat-diag_flat_exp)/norm_flat_exp
            FN_flat_exp[i] += (1/n_tries)*norm(FN_flat-diag_flat_exp)/norm_flat_exp
            PP_kr_flat_exp[i] += (1/n_tries)*norm(PP_kr_flat-diag_flat_exp)/norm_flat_exp
            PP_ch_flat_exp[i] += (1/n_tries)*norm(PP_ch_flat-diag_flat_exp)/norm_flat_exp
            PP_rem_flat_exp[i] += (1/n_tries)*norm(PP_rem_flat-diag_flat_exp)/norm_flat_exp

            kr_step=GirardHutchinsonFunctionDiagonalExpNorm(A_step,range_test[i],deg).+1
            FN_step=funNyströmDiag(A_step,fexp_norm,range_test[i]).+1
            PP_kr_step=funDiagPPExp(A_step,fexp_norm,range_test[i],deg).+1
            int_step=(0.001,1.0)
            PP_ch_step=ChebyshevfunDiagPP(A_step,fexp_norm_mat,fexp_norm,range_test[i],deg,int_step).+1
            PP_rem_step=RemezfunDiagPP(A_step,fexp_norm_mat,fexp_norm,range_test[i],deg,int_step).+1

            Krylov_error_step_exp[i] += (1/n_tries)*norm(kr_step-diag_step_exp)/norm_step_exp
            FN_step_exp[i] += (1/n_tries)*norm(FN_step-diag_step_exp)/norm_step_exp
            PP_kr_step_exp[i] += (1/n_tries)*norm(PP_kr_step-diag_step_exp)/norm_step_exp
            PP_ch_step_exp[i] += (1/n_tries)*norm(PP_ch_step-diag_step_exp)/norm_step_exp
            PP_rem_step_exp[i] += (1/n_tries)*norm(PP_rem_step-diag_step_exp)/norm_step_exp

            kr_poly=GirardHutchinsonFunctionDiagonalExpNorm(A_poly,range_test[i],deg).+1
            FN_poly=funNyströmDiag(A_poly,fexp_norm,range_test[i]).+1
            PP_kr_poly=funDiagPPExp(A_poly,fexp_norm,range_test[i],deg).+1
            int_poly=(0.000001,1.0)
            PP_ch_poly=ChebyshevfunDiagPP(A_poly,fexp_norm_mat,fexp_norm,range_test[i],deg,int_poly).+1
            PP_rem_poly=RemezfunDiagPP(A_poly,fexp_norm_mat,fexp_norm,range_test[i],deg,int_poly).+1

            Krylov_error_poly_exp[i] += (1/n_tries)*norm(kr_poly-diag_poly_exp)/norm_poly_exp
            FN_poly_exp[i] += (1/n_tries)*norm(FN_poly-diag_poly_exp)/norm_poly_exp
            PP_kr_poly_exp[i] += (1/n_tries)*norm(PP_kr_poly-diag_poly_exp)/norm_poly_exp
            PP_ch_poly_exp[i] += (1/n_tries)*norm(PP_ch_poly-diag_poly_exp)/norm_poly_exp
            PP_rem_poly_exp[i] += (1/n_tries)*norm(PP_rem_poly-diag_poly_exp)/norm_poly_exp

            kr_exp=GirardHutchinsonFunctionDiagonalExpNorm(A_exp,range_test[i],deg).+1
            FN_exp=funNyströmDiag(A_exp,fexp_norm,range_test[i]).+1
            PP_kr_exp=funDiagPPExp(A_exp,fexp_norm,range_test[i],deg).+1
            int_exp=(0.000001,1.0)
            PP_ch_exp=ChebyshevfunDiagPP(A_exp,fexp_norm_mat,fexp_norm,range_test[i],deg,int_exp).+1
            PP_rem_exp=RemezfunDiagPP(A_exp,fexp_norm_mat,fexp_norm,range_test[i],deg,int_exp).+1

            Krylov_error_exp_exp[i] += (1/n_tries)*norm(kr_exp-diag_exp_exp)/norm_exp_exp
            FN_exp_exp[i] += (1/n_tries)*norm(FN_exp-diag_exp_exp)/norm_exp_exp
            PP_kr_exp_exp[i] += (1/n_tries)*norm(PP_kr_exp-diag_exp_exp)/norm_exp_exp
            PP_ch_exp_exp[i] += (1/n_tries)*norm(PP_ch_exp-diag_exp_exp)/norm_exp_exp
            PP_rem_exp_exp[i] += (1/n_tries)*norm(PP_rem_exp-diag_exp_exp)/norm_exp_exp
    end
end

## plot
plot1 = scatter(range_test,Krylov_error_flat_exp,label="", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,legend=:topright,xticks=0:50:300,ylims=(10^(-2),10^0),yticks=[10^(-2), 10^(-1),10^0],size=(1250,400),margin=10pt)
plot!(plot1,range_test,Krylov_error_flat_exp,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,FN_flat_exp,label="",color="green",yaxis=:log)
plot!(plot1,range_test,FN_flat_exp,label="",color="green",yaxis=:log)
scatter!(plot1,range_test,PP_ch_flat_exp,label="",color="yellow",yaxis=:log )
plot!(plot1,range_test,PP_ch_flat_exp,label="",color="yellow",yaxis=:log)
scatter!(plot1,range_test,PP_rem_flat_exp,label="",color="orange",yaxis=:log )
plot!(plot1,range_test,PP_rem_flat_exp,label="",color="orange",yaxis=:log)
scatter!(plot1,range_test,PP_kr_flat_exp,label="",color="blue",yaxis=:log )
plot!(plot1,range_test,PP_kr_flat_exp,label="",color="blue",yaxis=:log)

plot2 = scatter(range_test,Krylov_error_step_exp,label="GH+Krylov", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:300,legend=:outertopright,size=(1250,400),ylims=(10^(-3.5),10^0),yticks=[10^(-3), 10^(-2), 10^(-1),10^0],margin=10pt)
plot!(plot2,range_test,Krylov_error_step_exp,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,FN_step_exp,label="funNyström",color="green",yaxis=:log)
plot!(plot2,range_test,FN_step_exp,label="",color="green",yaxis=:log)
scatter!(plot2,range_test,PP_ch_step_exp,label="funDiag++ Chebyshev",color="yellow",yaxis=:log )
plot!(plot2,range_test,PP_ch_step_exp,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,PP_rem_step_exp,label="funDiag++ Remez",color="orange",yaxis=:log )
plot!(plot2,range_test,PP_rem_step_exp,label="",color="orange",yaxis=:log)
scatter!(plot2,range_test,PP_kr_step_exp,label="funDiag++ Krylov",color="blue",yaxis=:log )
plot!(plot2,range_test,PP_kr_step_exp,label="",color="blue",yaxis=:log)

plot3 = scatter(range_test,Krylov_error_poly_exp,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:300,legend=:topright,size=(1250,400),ylims=(10^(-9),10^0),yticks=[10^(-9), 10^(-6), 10^(-3),10^0],margin=10pt)
plot!(plot3,range_test,Krylov_error_poly_exp,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,FN_poly_exp,label="",color="green",yaxis=:log)
plot!(plot3,range_test,FN_poly_exp,label="",color="green",yaxis=:log)
scatter!(plot3,range_test,PP_ch_poly_exp,label="",color="yellow",yaxis=:log )
plot!(plot3,range_test,PP_ch_poly_exp,label="",color="yellow",yaxis=:log)
scatter!(plot3,range_test,PP_rem_poly_exp,label="",color="orange",yaxis=:log )
plot!(plot3,range_test,PP_rem_poly_exp,label="",color="orange",yaxis=:log)
scatter!(plot3,range_test,PP_kr_poly_exp,label="",color="blue",yaxis=:log )
plot!(plot3,range_test,PP_kr_poly_exp,label="",color="blue",yaxis=:log)

plot4 = scatter(range_test,Krylov_error_exp_exp,label="", title="Exp", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:300,ylims=(10^(-16),10^0),yticks=[10^(-16), 10^(-8),10^0],legend=:topright,size=(1250,400),margin=10pt)
plot!(plot4,range_test,Krylov_error_exp_exp,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,FN_exp_exp,label="",color="green",yaxis=:log)
plot!(plot4,range_test,FN_exp_exp,label="",color="green",yaxis=:log)
scatter!(plot4,range_test,PP_ch_exp_exp,label="",color="yellow",yaxis=:log)
plot!(plot4,range_test,PP_ch_exp_exp,label="",color="yellow",yaxis=:log)
scatter!(plot4,range_test,PP_rem_exp_exp,label="",color="orange",yaxis=:log )
plot!(plot4,range_test,PP_rem_exp_exp,label="",color="orange",yaxis=:log)
scatter!(plot4,range_test,PP_kr_exp_exp,label="",color="blue",yaxis=:log )
plot!(plot4,range_test,PP_kr_exp_exp,label="",color="blue",yaxis=:log)

l = @layout [a b;c d]
plot(plot1, plot2, plot3, plot4, layout = l,size=(1000,500))





















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
diag_exp_log= diag(A_exp_log)
diag_step_log = diag(A_step_log)

norm_flat_log = norm(diag_flat_log)
norm_poly_log = norm(diag_poly_log)
norm_exp_log = norm(diag_exp_log)
norm_step_log = norm(diag_step_log);



## Simple plot
deg = 25
range_test = 20:20:300
n_tries = 5
lim_q = length(range_test)

# Chebyshev_error_flat_log = zeros(lim_q)
# Remez_error_flat_log = zeros(lim_q)
Krylov_error_flat_log = zeros(lim_q)
FN_flat_log = zeros(lim_q)
PP_kr_flat_log = zeros(lim_q)
PP_ch_flat_log = zeros(lim_q)
PP_rem_flat_log = zeros(lim_q)

Krylov_error_step_log = zeros(lim_q)
FN_step_log = zeros(lim_q)
PP_kr_step_log = zeros(lim_q)
PP_ch_step_log = zeros(lim_q)
PP_rem_step_log = zeros(lim_q)

Krylov_error_poly_log = zeros(lim_q)
FN_poly_log = zeros(lim_q)
PP_kr_poly_log = zeros(lim_q)
PP_ch_poly_log = zeros(lim_q)
PP_rem_poly_log = zeros(lim_q)

Krylov_error_exp_log = zeros(lim_q)
FN_exp_log = zeros(lim_q)
PP_kr_exp_log = zeros(lim_q)
PP_ch_exp_log = zeros(lim_q)
PP_rem_exp_log = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            kr_flat=GirardHutchinsonFunctionDiagonalKrylovCustom(A_flat,range_test[i],flogmat,deg)
            FN_flat=funNyströmDiag(A_flat,flog,range_test[i])
            PP_kr_flat=KrylovfunDiagPPcustom(A_flat, flogmat,flog,range_test[i],deg)
            int_flat=(1.0,3.0)
            PP_ch_flat=ChebyshevfunDiagPP(A_flat,flogmat,flog,range_test[i],deg,int_flat)
            PP_rem_flat=RemezfunDiagPP(A_flat,flogmat,flog,range_test[i],deg,int_flat)

            Krylov_error_flat_log[i] += (1/n_tries)*norm(kr_flat-diag_flat_log)/norm_flat_log
            FN_flat_log[i] += (1/n_tries)*norm(FN_flat-diag_flat_log)/norm_flat_log
            PP_kr_flat_log[i] += (1/n_tries)*norm(PP_kr_flat-diag_flat_log)/norm_flat_log
            PP_ch_flat_log[i] += (1/n_tries)*norm(PP_ch_flat-diag_flat_log)/norm_flat_log
            PP_rem_flat_log[i] += (1/n_tries)*norm(PP_rem_flat-diag_flat_log)/norm_flat_log

            kr_step=GirardHutchinsonFunctionDiagonalKrylovCustom(A_step,range_test[i],flogmat,deg)
            FN_step=funNyströmDiag(A_step,flog,range_test[i])
            PP_kr_step=KrylovfunDiagPPcustom(A_step, flogmat,flog,range_test[i],deg)
            int_step=(0.001,1.0)
            PP_ch_step=ChebyshevfunDiagPP(A_step,flogmat,flog,range_test[i],deg,int_step)
            PP_rem_step=RemezfunDiagPP(A_step,flogmat,flog,range_test[i],deg,int_step)

            Krylov_error_step_log[i] += (1/n_tries)*norm(kr_step-diag_step_log)/norm_step_log
            FN_step_log[i] += (1/n_tries)*norm(FN_step-diag_step_log)/norm_step_log
            PP_kr_step_log[i] += (1/n_tries)*norm(PP_kr_step-diag_step_log)/norm_step_log
            PP_ch_step_log[i] += (1/n_tries)*norm(PP_ch_step-diag_step_log)/norm_step_log
            PP_rem_step_log[i] += (1/n_tries)*norm(PP_rem_step-diag_step_log)/norm_step_log

            kr_poly=GirardHutchinsonFunctionDiagonalKrylovCustom(A_poly,range_test[i],flogmat,deg)
            FN_poly=funNyströmDiag(A_poly,flog,range_test[i])
            PP_kr_poly=KrylovfunDiagPPcustom(A_poly, flogmat,flog,range_test[i],deg)
            int_poly=(0.000001,1.0)
            PP_ch_poly=ChebyshevfunDiagPP(A_poly,flogmat,flog,range_test[i],deg,int_poly)
            PP_rem_poly=RemezfunDiagPP(A_poly,flogmat,flog,range_test[i],deg,int_poly)

            Krylov_error_poly_log[i] += (1/n_tries)*norm(kr_poly-diag_poly_log)/norm_poly_log
            FN_poly_log[i] += (1/n_tries)*norm(FN_poly-diag_poly_log)/norm_poly_log
            PP_kr_poly_log[i] += (1/n_tries)*norm(PP_kr_poly-diag_poly_log)/norm_poly_log
            PP_ch_poly_log[i] += (1/n_tries)*norm(PP_ch_poly-diag_poly_log)/norm_poly_log
            PP_rem_poly_log[i] += (1/n_tries)*norm(PP_rem_poly-diag_poly_log)/norm_poly_log

            kr_exp=GirardHutchinsonFunctionDiagonalKrylovCustom(A_exp,range_test[i],flogmat,deg)
            FN_exp=funNyströmDiag(A_exp,flog,range_test[i])
            PP_kr_exp=KrylovfunDiagPPcustom(A_exp, flogmat,flog,range_test[i],deg)
            int_exp=(0.000001,1.0)
            PP_ch_exp=ChebyshevfunDiagPP(A_exp,flogmat,flog,range_test[i],deg,int_exp)
            PP_rem_exp=RemezfunDiagPP(A_exp,flogmat,flog,range_test[i],deg,int_exp)

            Krylov_error_exp_log[i] += (1/n_tries)*norm(kr_exp-diag_exp_log)/norm_exp_log
            FN_exp_log[i] += (1/n_tries)*norm(FN_exp-diag_exp_log)/norm_exp_log
            PP_kr_exp_log[i] += (1/n_tries)*norm(PP_kr_exp-diag_exp_log)/norm_exp_log
            PP_ch_exp_log[i] += (1/n_tries)*norm(PP_ch_exp-diag_exp_log)/norm_exp_log
            PP_rem_exp_log[i] += (1/n_tries)*norm(PP_rem_exp-diag_exp_log)/norm_exp_log
    end
end

## plot
plot1 = scatter(range_test,Krylov_error_flat_log,label="", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,legend=:topright,xticks=0:50:600,ylims=(10^(-2.5),10^0),yticks=[10^(-2), 10^(-1), 10^0],size=(1250,400),margin=10pt)
plot!(plot1,range_test,Krylov_error_flat_log,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,FN_flat_log,label="",color="green",yaxis=:log)
plot!(plot1,range_test,FN_flat_log,label="",color="green",yaxis=:log)
scatter!(plot1,range_test,PP_ch_flat_log,label="",color="yellow",yaxis=:log )
plot!(plot1,range_test,PP_ch_flat_log,label="",color="yellow",yaxis=:log)
scatter!(plot1,range_test,PP_rem_flat_log,label="",color="orange",yaxis=:log )
plot!(plot1,range_test,PP_rem_flat_log,label="",color="orange",yaxis=:log)
scatter!(plot1,range_test,PP_kr_flat_log,label="",color="blue",yaxis=:log )
plot!(plot1,range_test,PP_kr_flat_log,label="",color="blue",yaxis=:log)

plot2 = scatter(range_test,Krylov_error_step_log,label="GH+Krylov", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:600,legend=:outertopright,size=(1250,400),ylims=(10^(-2.5),2*10^0),margin=10pt)
plot!(plot2,range_test,Krylov_error_step_log,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,FN_step_log,label="funNyström",color="green",yaxis=:log)
plot!(plot2,range_test,FN_step_log,label="",color="green",yaxis=:log)
scatter!(plot2,range_test,PP_ch_step_log,label="funDiag++ Chebyshev",color="yellow",yaxis=:log )
plot!(plot2,range_test,PP_ch_step_log,label="",color="blue",yaxis=:log)
scatter!(plot2,range_test,PP_rem_step_log,label="funDiag++ Remez",color="orange",yaxis=:log )
plot!(plot2,range_test,PP_rem_step_log,label="",color="orange",yaxis=:log)
scatter!(plot2,range_test,PP_kr_step_log,label="funDiag++ Krylov",color="blue",yaxis=:log )
plot!(plot2,range_test,PP_kr_step_log,label="",color="blue",yaxis=:log)

plot3 = scatter(range_test,Krylov_error_poly_log,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:600,legend=:topright,size=(1250,400),ylims=(10^(-5),10^1),margin=10pt)
plot!(plot3,range_test,Krylov_error_poly_log,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,FN_poly_log,label="",color="green",yaxis=:log)
plot!(plot3,range_test,FN_poly_log,label="",color="green",yaxis=:log)
scatter!(plot3,range_test,PP_ch_poly_log,label="",color="yellow",yaxis=:log )
plot!(plot3,range_test,PP_ch_poly_log,label="",color="yellow",yaxis=:log)
scatter!(plot3,range_test,PP_rem_poly_log,label="",color="orange",yaxis=:log )
plot!(plot3,range_test,PP_rem_poly_log,label="",color="orange",yaxis=:log)
scatter!(plot3,range_test,PP_kr_poly_log,label="",color="blue",yaxis=:log )
plot!(plot3,range_test,PP_kr_poly_log,label="",color="blue",yaxis=:log)

plot4 = scatter(range_test,Krylov_error_exp_log,label="", title="Exp", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",yaxis=:log,xticks=0:50:600,ylims=(10^(-16),10^1),yticks=[10^(-15), 10^(-10), 10^(-5), 10^0],legend=:topright,size=(1250,400),margin=10pt)
plot!(plot4,range_test,Krylov_error_exp_log,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,FN_exp_log,label="",color="green",yaxis=:log)
plot!(plot4,range_test,FN_exp_log,label="",color="green",yaxis=:log)
scatter!(plot4,range_test,PP_ch_exp_log,label="",color="yellow",yaxis=:log)
plot!(plot4,range_test,PP_ch_exp_log,label="",color="yellow",yaxis=:log)
scatter!(plot4,range_test,PP_rem_exp_log,label="",color="orange",yaxis=:log )
plot!(plot4,range_test,PP_rem_exp_log,label="",color="orange",yaxis=:log)
scatter!(plot4,range_test,PP_kr_exp_log,label="",color="blue",yaxis=:log)
plot!(plot4,range_test,PP_kr_exp_log,label="",color="blue",yaxis=:log)

l = @layout [a b;c d]
plot(plot1, plot2, plot3, plot4, layout = l,size=(1000,500))
