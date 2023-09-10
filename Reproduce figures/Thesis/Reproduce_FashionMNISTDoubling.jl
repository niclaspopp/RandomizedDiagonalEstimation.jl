using MLDatasets
using ImageCore
using Krylov
using LinearAlgebra
using MAT
using RandomizedDiagonalEstimation
## load custom functions for this application
include("./DoublingStrategy.jl")
## load data
labels=matread("Applications/MNIST/FashinMNIST10000labels.mat")["labels"]
labels=vec(labels);
Data=matread("Applications/MNIST/FashinMNIST10000.mat")["Data"];
ATA=matread("Applications/MNIST/FashinMNIST10000ATA.mat")["ATA"];
b_cg=matread("Applications/MNIST/FashinMNIST10000b_cg.mat")["b_cg"];
res_sol=matread("Applications/MNIST/FashinMNIST10000var.mat")["var"];
## Construct preconditioners
diag_est=RandomizedDiagonalEstimation.EstimateDiagonal(ATA,:DiagPP, :queries, :Gaussian, true,maxqueries=300)
Rand_Precond = Diagonal(diag_est.^(-1));
## Perfrom experiments


e1,v1=GHDIPreCG_storevar(ATA,25,1000,Rand_Precond)
e2,v2=GHDIPreCG_storevar(ATA,50,1000,Rand_Precond)
e3,v3=GHDIPreCG_storevar(ATA,100,1000,Rand_Precond)
e4,v4=GHDIPreCG_storevar(ATA,200,1000,Rand_Precond)
e5,v5=GHDIPreCG_storevar(ATA,400,1000,Rand_Precond)
e6,v6=GHDIPreCG_storevar(ATA,800,1000,Rand_Precond)
e7,v7=GHDIPreCG_storevar(ATA,1600,1000,Rand_Precond)


var_ests=[v1,v2,v3,v4,v5,v6,v7]./norm(e7)
n_queries = [25,50,100,200,400,800,1600]
xtickers=n_queries[3:end]
## plot results

scatter(n_queries,var_ests,yaxis=:log,ylabel=L"\widehat{\epsilon}",xlabel="Number of test vectors",label="",xticks=xtickers,color=:blue)
plot!(n_queries,var_ests,color=:blue,label="")
hline!([0.05],label=L"5\%\textrm{ estimated relative error}",linestyle=:dot,color=:red,legend=:topright,grid = false,size=(400,300))
