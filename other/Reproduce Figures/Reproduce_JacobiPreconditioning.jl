using DelimitedFiles, Random, Statistics, Roots, LinearAlgebra
using Distributions, Statistics, Plots, ToeplitzMatrices
using RandomizedLinAlg, Measures, MatrixDepot, Random
using MAT, SparseArrays, Optim, BandedMatrices
using LinearSolve, RandomizedPreconditioners
using LaTeXStrings, RandomizedLinAlg
pgfplotsx()
using RandomizedDiagonalEstimation
## CG and CG Precond
function cg_complex(A,b,m)
    n=length(b);
    x=zeros(Complex{Float64},m+1,n);
    r=zeros(Complex{Float64},m+1,n);
    p=zeros(Complex{Float64},m+1,n);
    α=zeros(Complex{Float64},m);
    β=zeros(Complex{Float64},m);

    x[1,:].=0;
    r[1,:]=b;
    p[1,:]=r[1,:];

    for i in 1:m
        # !important: perform the matrix multiplication beforhand to decrease the cost! #
        temp=(A*p[i,:])
        α[i] = (r[i,:]'*r[i,:])/(p[i,:]'*temp)
        x[i+1,:] = x[i,:] + α[i]*p[i,:]
        r[i+1,:] = r[i,:] - α[i]*temp
        β[i] = (r[i+1,:]'*r[i+1,:])/(r[i,:]'*r[i,:])
        p[i+1,:] = r[i+1,:] + β[i]*p[i,:]
    end
    return x[m+1,:]
end

function cg_preconditioned_complex(A,Sinv,b,m)
    n=length(b);
    x=zeros(Complex{Float64},m+1,n);
    r=zeros(Complex{Float64},m+1,n);
    p=zeros(Complex{Float64},m+1,n);
    α=zeros(Complex{Float64},m);
    β=zeros(Complex{Float64},m);

    x[1,:].=0;
    r[1,:]=b;
    p[1,:]=Sinv*r[1,:];

    for i in 1:m
        # !important: perform the matrix multiplication beforhand to decrease the cost! #
        temp=A*p[i,:]
        α[i] = (r[i,:]'*Sinv*r[i,:])/(p[i,:]'*temp)
        x[i+1,:] = x[i,:] + α[i]*p[i,:]
        r[i+1,:] = r[i,:] - α[i]*temp
        β[i] = (r[i+1,:]'*Sinv*r[i+1,:])/(r[i,:]'*Sinv*r[i,:])
        p[i+1,:] = Sinv*r[i+1,:] + β[i]*p[i,:]
    end
    return x[m+1,:]
end

## Constrtuct the Example Matrix
n = 1000
A = zeros(Float64,n,n)
for i=1:n
    A[i,i]=0.5+sqrt(i)
    for j=i+1:n
        if abs(i-j)==100
            A[i,j]=1
            A[j,i]=1
        end
    end
end

b=ones(Float64,n);

Precond = Diagonal(diag(A).^(-1));
diag_est=EstimateDiagonal(A,:GirardHutchinson, :queries, :Gaussian, true,maxqueries=10);
Rand_Precond = Diagonal(diag_est.^(-1));
prob = LinearProblem(A, b)
sol = solve(prob, IterativeSolversJL_CG(),maxiters=10,Pl=Rand_Precond);

## Test runs
ms = 1:1:40
# errors
err_cg = zeros(length(ms))
err_cg_jacobi = zeros(length(ms));
err_cg_randomizedjacobi = zeros(length(ms));

# run loop
for i in 1:length(ms)
    #CG
    x = cg_complex(A,b,ms[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i])
    err_cg[i]=norm(A*x-b)

    # Jacobi Preconditioning
    x = cg_preconditioned_complex(A,Precond,b,ms[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i],Pl=Precond)
    err_cg_jacobi[i]=norm(A*x-b)

    # Randomized Jacobi Preconditioning
    x = cg_preconditioned_complex(A,Rand_Precond,b,ms[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i],Pl=Rand_Precond)
    err_cg_randomizedjacobi[i]=norm(A*x-b)

end
##
A_2=matread("Jacobi Preconditioning/FashinMNIST10000ATA.mat")["ATA"];
n_2 = size(A_2)[1]
b_2=ones(Float64,n_2);

Precond_2=Diagonal(diag(A_2).^(-1));
diag_est_2=EstimateDiagonal(A_2,:DiagPP, :queries, :Gaussian, true,maxqueries=300);
Rand_Precond_2=Diagonal(diag_est_2.^(-1));

norm(diag(A_2)-diag_est_2)/norm(diag_est_2)
##convergence analysis
ms_2 = 1:20:401
err_cg_2 = zeros(length(ms_2))
err_cg_jacobi_2 = zeros(length(ms_2));
err_cg_randomizedjacobi_2 = zeros(length(ms_2));

for i in 1:length(ms_2)

    #CG
    x = cg_complex(A_2,b_2,ms_2[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i])
    err_cg_2[i]=norm(A_2*x-b_2)

    # Jacobi Preconditioning
    x = cg_preconditioned_complex(A_2,Precond_2,b_2,ms_2[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i],Pl=Precond)
    err_cg_jacobi_2[i]=norm(A_2*x-b_2)

    # Randomized Jacobi Preconditioning
    x = cg_preconditioned_complex(A_2,Rand_Precond_2,b_2,ms_2[i])#solve(prob, IterativeSolversJL_CG(),maxiters=ms[i],Pl=Rand_Precond)
    err_cg_randomizedjacobi_2[i]=norm(A_2*x-b_2)

end

## Plot
plot1=scatter(ms,err_cg,label="", title="Trefethen Test Matrix", legend=:topright,xlabel="Iteration",ylabel=L"\|r_n\|_2",yaxis=:log,color="green")
plot!(plot1,ms,err_cg,label="",color="green")
scatter!(plot1,ms,err_cg_jacobi,label="",color="blue")
plot!(plot1,ms,err_cg_jacobi,label="",color="blue")
scatter!(plot1,ms,err_cg_randomizedjacobi,label="",color="red")
plot!(plot1,ms,err_cg_randomizedjacobi,label="",color="red")

plot2=scatter(ms_2,err_cg_2,label="No Preconditioning", title="OLS Fashion MNIST Matrix", legend=:bottomleft,xlabel="Iteration",ylabel=L"\|r_n\|_2",yaxis=:log,color="green")
plot!(plot2,ms_2,err_cg_2,label="",color="green")
scatter!(plot2,ms_2,err_cg_jacobi_2,label="Jacobi Preconditioning",color="blue")
plot!(plot2,ms_2,err_cg_jacobi_2,label="",color="blue")
scatter!(plot2,ms_2,err_cg_randomizedjacobi_2,label="Randomized Jacobi Preconditioning",color="red")
plot!(plot2,ms_2,err_cg_randomizedjacobi_2,label="",color="red")

l = @layout [a b]
plot(plot1, plot2, layout = l,size=(800,300))
