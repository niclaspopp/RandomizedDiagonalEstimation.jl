using RandomizedPreconditioners
using DataFrames, StatsBase, Plots, Distributions, Statistics, LinearAlgebra
using PGFPlotsX, LaTeXStrings, RandomizedLinAlg, Random, Measures
using IterativeSolvers
using RandomizedDiagonalEstimation
pgfplotsx()

## Define custom functions
# These functions are not part of the package. The reason is that the package integrates with existing Julia packages. This is to refrain from re-implementanting function that exists in specified packages
# Yet for these experiments we found that some of these packages that the arnoldi method part of ExponentialUtilities.jl is unstable in some cases such that we use our own implementation

function gs_modified(Q,w,k)
    # For Loop
    y=w
    h=zeros(k)
    for i in 1:k
        h[i]=Q[:,i]'y
        y=y-Q[:,i]*h[i]
    end
    return h,norm(y),y
end

function arnoldi(A,b,m)

    n=length(b);
    Q=zeros(n,m+1);
    H=zeros(m+1,m);
    Q[:,1]=b/norm(b)

    for k=1:m
        w=A*Q[:,k]; # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        h,β,z=gs_modified(Q,w,k);
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    H[H.<5*10^(-16)].=0
    return Q,H
end

function KrylovMatfuncCustom(A,b,f,m_Kr)
    V,H=arnoldi(A,b./norm(b),m_Kr);
    result=vec(V[:,1:m_Kr]*(f(H[1:m_Kr,:])*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    return result
end

function GirardHutchinsonFunctionDiagonalKrylovCustom(A,s,f,m_Kr,dist=:Gaussian,normal=true,orthogonalize=false,testmatrix=nothing)
    # A: input matrix, s: number of samples, p: degree of the polynomial approximation, dist: distribution for random vectors, normalization: select if normalization should be performed or not

    # get sizes of the matrix
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Sample random test matrix
    if dist==:Rademacher
        d = Bernoulli(.5)
        Omega=2*rand(d,n,s).-1
    elseif dist==:Gaussian
        Omega=randn(n,s)
    elseif dist==:custom
        Omega=testmatrix
    else
        print("Unknown distribution")
    end
    # orthogonalize test matrix
    if orthogonalize==true
        Q,R=qr(Omega)
        Om=Matrix(Q)
    else
        Om=Omega
    end
    # Perform action based on Krylov subspace
    Action=zeros(n,s)
    for j=1:s
        # Compute Krylov subspace
        Action[:,j]=KrylovMatfuncCustom(A,Om[:,j],f,m_Kr)
    end
    #print(Action - Ainv*Om)
    Action=Action.*Om
	#Calculate normalization
	if normal==true
		V = Om.*Om
		normalization =  sum(V,dims=2)
	else
		normalization = s*ones(n)
	end
	# Calculate the final result
	result = sum(Action,dims=2)./normalization

	return vec(result)
end

function nystrom(A, r, q)
    # Implementation of the Nystrom approximation

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    Omega, R = qr(randn(n, r))
    Omega = Matrix(Omega)
    iteration = 0

    # Subspace iteration
    while iteration < q-1
        Omega, R = qr(A*Omega)
        Omega = Matrix(Omega)
        iteration += 1
    end

    # Compute matrix products with A
    Y = A*Omega

    # Regularization
    U, S, Vt = svd(Omega'*Y)
    S[S .< 5e-16*S[1,1]] .= 0
    B = Y * (Vt' * pinv(diagm(sqrt.(S))) * Vt)
    U, Shat, Vhatt = svd(B)
    S = Shat.^2
    return U, S
end

function KrylovfunDiagPP(A, fAfun,fscalar,s_in,deg,q=2)

    r = Int(round(s_in/2))
    m = s_in-r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	U, S = nystrom(A, r, q)
	N = U*diagm(fscalar.(S))*U'
	diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega=2*rand(d,n,m).-1
    Action=zeros(n,m)
    for j=1:m
        # Compute Krylov subspace
        Action[:,j]=KrylovMatfuncCustom(A,Omega[:,j],fAfun,deg)
    end
    # Y = U' * Omega
    diag2=zeros(n)
    for j=1:m
        # Compute Krylov subspace
        diag2+=(Action[:,j].*Omega[:,j]-(N*Omega[:,j]).*Omega[:,j]) / m
    end
    #t2 = (tr(Omega' * Action) - tr(Y' * diagm(fscalar.(S)) * Y)) / m

    # Return result
    return diag1+diag2
end

function funNyströmDiag(A, fAfun,fscalar,s_in,deg,q=2)
    r = s_in
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	U, S = nystrom(A, r, q)
    N = U*diagm(fscalar.(S))*U'
	diag1=diag(N)
    return diag1
end

function funNyströmDiag2(A, fAfun,fscalar,s_in,deg)
	NS2=NystromSketch(A, s_in, s_in)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
	diag1=diag(N)
end

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

#define normalized exponential functions
fexp = x->exp(x)
fexp_norm = x->exp(x).-1
fexp_norm_mat = x->exp(x).-Matrix(I,size(x))

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

##
fdkr=GirardHutchinsonFunctionDiagonalKrylovCustom(A_exp,300,fexp,26)
fdpp=KrylovfunDiagPP(A_exp, fexp_norm_mat,fexp_norm,300,26,5).+1
fn = funNyströmDiag(A_exp, fexp_norm_mat,fexp_norm,300,26,5).+1
# fn2 = funNyströmDiag2(A_exp, fexp_norm_mat,fexp_norm,300,26).+1

norm(fdkr-diag_exp_exp)/norm_exp_exp
norm(fdpp-diag_exp_exp)/norm_exp_exp
norm(fn-diag_exp_exp)/norm_exp_exp
# norm(fn2-diag_exp_exp)/norm_exp_exp
## Perform experiments
n_tries = 11
range_test = 20:20:300
lim_q = length(range_test)
deg = 25
q_test=6

Exact_fA_error_flat = zeros(lim_q)
Exact_fA_error_poly = zeros(lim_q)
Exact_fA_error_exp = zeros(lim_q)
Exact_fA_error_step = zeros(lim_q)

Chebyshev_error_flat = zeros(lim_q)
Chebyshev_error_poly = zeros(lim_q)
Chebyshev_error_exp = zeros(lim_q)
Chebyshev_error_step = zeros(lim_q)

Remez_error_flat = zeros(lim_q)
Remez_error_poly = zeros(lim_q)
Remez_error_exp = zeros(lim_q)
Remez_error_step = zeros(lim_q)

Krylov_error_flat = zeros(lim_q)
Krylov_error_poly = zeros(lim_q)
Krylov_error_exp = zeros(lim_q)
Krylov_error_step = zeros(lim_q)

funDiagPP_error_flat = zeros(lim_q)
funDiagPP_error_poly = zeros(lim_q)
funDiagPP_error_exp = zeros(lim_q)
funDiagPP_error_step = zeros(lim_q)

funNy_error_flat = zeros(lim_q)
funNy_error_poly = zeros(lim_q)
funNy_error_exp = zeros(lim_q)
funNy_error_step = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
			GH_exact_exp_flat=EstimateDiagonal(A_flat_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_flat=EstimateFunctionDiagonal(A_flat,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_flat=EstimateFunctionDiagonal(A_flat,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_flat=GirardHutchinsonFunctionDiagonalKrylovCustom(A_flat,range_test[i],fexp,deg+1)
			# Add try-catck block to handle LAPACK exceptions
			fundiagpp_flat=zeros(n)
			funNy_flat=zeros(n)
			try
				fundiagpp_flat=KrylovfunDiagPP(A_flat, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_flat=funNyströmDiag(A_flat, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			catch
				fundiagpp_flat=KrylovfunDiagPP(A_flat, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_flat=funNyströmDiag(A_flat, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			end


            Exact_fA_error_flat[i] += (1/n_tries)*norm(GH_exact_exp_flat-diag_flat_exp)/norm_flat_exp
            Chebyshev_error_flat[i] += (1/n_tries)*norm(ch_flat-diag_flat_exp)/norm_flat_exp
            Remez_error_flat[i] += (1/n_tries)*norm(rem_flat-diag_flat_exp)/norm_flat_exp
            Krylov_error_flat[i] += (1/n_tries)*norm(kr_flat-diag_flat_exp)/norm_flat_exp
            funDiagPP_error_flat[i] += (1/n_tries)*norm(fundiagpp_flat-diag_flat_exp)/norm_flat_exp
			funNy_error_flat[i] += (1/n_tries)*norm(funNy_flat-diag_flat_exp)/norm_flat_exp

			GH_exact_exp_step=EstimateDiagonal(A_step_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_step=EstimateFunctionDiagonal(A_step,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_step=EstimateFunctionDiagonal(A_step,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_step=GirardHutchinsonFunctionDiagonalKrylovCustom(A_step,range_test[i],fexp,deg+1)
			fundiagpp_step=zeros(n)
			funNy_step=zeros(n)
			try
				fundiagpp_step=KrylovfunDiagPP(A_step, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_step=funNyströmDiag(A_step, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			catch
				fundiagpp_step=KrylovfunDiagPP(A_step, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_step=funNyströmDiag(A_step, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			end

			Exact_fA_error_step[i] += (1/n_tries)*norm(GH_exact_exp_step-diag_step_exp)/norm_step_exp
			Chebyshev_error_step[i] += (1/n_tries)*norm(ch_step-diag_step_exp)/norm_step_exp
			Remez_error_step[i] += (1/n_tries)*norm(rem_step-diag_step_exp)/norm_step_exp
			Krylov_error_step[i] += (1/n_tries)*norm(kr_step-diag_step_exp)/norm_step_exp
			funDiagPP_error_step[i] += (1/n_tries)*norm(fundiagpp_step-diag_step_exp)/norm_step_exp
			funNy_error_step[i] += (1/n_tries)*norm(funNy_step-diag_step_exp)/norm_step_exp

			GH_exact_exp_poly=EstimateDiagonal(A_poly_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_poly=EstimateFunctionDiagonal(A_poly,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_poly=EstimateFunctionDiagonal(A_poly,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_poly=GirardHutchinsonFunctionDiagonalKrylovCustom(A_poly,range_test[i],fexp,deg+1)
			fundiagpp_poly=zeros(n)
			funNy_poly=zeros(n)
			try
				fundiagpp_poly=KrylovfunDiagPP(A_poly, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_poly=funNyströmDiag(A_poly, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			catch
				fundiagpp_poly=KrylovfunDiagPP(A_poly, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_poly=funNyströmDiag(A_poly, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			end


			Exact_fA_error_poly[i] += (1/n_tries)*norm(GH_exact_exp_poly-diag_poly_exp)/norm_poly_exp
			Chebyshev_error_poly[i] += (1/n_tries)*norm(ch_poly-diag_poly_exp)/norm_poly_exp
			Remez_error_poly[i] += (1/n_tries)*norm(rem_poly-diag_poly_exp)/norm_poly_exp
			Krylov_error_poly[i] += (1/n_tries)*norm(kr_poly-diag_poly_exp)/norm_poly_exp
			funDiagPP_error_poly[i] += (1/n_tries)*norm(fundiagpp_poly-diag_poly_exp)/norm_poly_exp
			funNy_error_poly[i] += (1/n_tries)*norm(funNy_poly-diag_poly_exp)/norm_poly_exp

			GH_exact_exp_exp=EstimateDiagonal(A_exp_exp,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_exp=EstimateFunctionDiagonal(A_exp,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_exp=EstimateFunctionDiagonal(A_exp,fexp,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_exp=GirardHutchinsonFunctionDiagonalKrylovCustom(A_exp,range_test[i],fexp,deg+1)
			fundiagpp_exp=zeros(n)
			funNy_exp=zeros(n)
			try
				fundiagpp_exp=KrylovfunDiagPP(A_exp, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_exp=funNyströmDiag(A_exp, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			catch
				fundiagpp_exp=KrylovfunDiagPP(A_exp, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
				funNy_exp=funNyströmDiag(A_exp, fexp_norm_mat,fexp_norm,range_test[i],deg+1,q_test).+1
			end


            Exact_fA_error_exp[i] += (1/n_tries)*norm(GH_exact_exp_exp-diag_exp_exp)/norm_exp_exp
            Chebyshev_error_exp[i] += (1/n_tries)*norm(ch_exp-diag_exp_exp)/norm_exp_exp
            Remez_error_exp[i] += (1/n_tries)*norm(rem_exp-diag_exp_exp)/norm_exp_exp
            Krylov_error_exp[i] += (1/n_tries)*norm(kr_exp-diag_exp_exp)/norm_exp_exp
            funDiagPP_error_exp[i] += (1/n_tries)*norm(fundiagpp_exp-diag_exp_exp)/norm_exp_exp
			funNy_error_exp[i] += (1/n_tries)*norm(funNy_exp-diag_exp_exp)/norm_exp_exp
    end
end
## Plot the results
#plot(1:10,1:10)
l = @layout [a b;c d]

plot1 = scatter(range_test,Exact_fA_error_flat,label="", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",markershape = :pentagon,yaxis=:log,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot1,range_test,Exact_fA_error_flat,label="",color="red",yaxis=:log)
scatter!(plot1,range_test,Chebyshev_error_flat,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot1,range_test,Chebyshev_error_flat,label="",color="green",yaxis=:log)
scatter!(plot1,range_test,Remez_error_flat,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!(plot1,range_test,Remez_error_flat,label="",color="yellow",yaxis=:log)
scatter!(plot1,range_test,Krylov_error_flat,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot1,range_test,Krylov_error_flat,label="",color="orange",yaxis=:log)
scatter!(plot1,range_test,funDiagPP_error_flat,label="",color="brown",markershape = :rect,yaxis=:log)
plot!(plot1,range_test,funDiagPP_error_flat,label="",color="brown",yaxis=:log)
scatter!(plot1,range_test,funNy_error_flat,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!(plot1,range_test,funNy_error_flat,label="",color="pink",yaxis=:log)


plot2 = scatter(range_test,Exact_fA_error_step,label=L"\textrm{GH }+f(\textbf{A})", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :circle,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot2,range_test,Exact_fA_error_step,label="",color="red",yaxis=:log)
scatter!(plot2,range_test,Chebyshev_error_step,label="funDiag Chebyshev",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot2,range_test,Chebyshev_error_step,label="",color="green",yaxis=:log)
scatter!(plot2,range_test,Remez_error_step,label="funDiag Remez",color="yellow",markershape = :diamond,yaxis=:log)
plot!(plot2,range_test,Remez_error_step,label="",color="yellow",yaxis=:log)
scatter!(plot2,range_test,Krylov_error_step,label="funDiag Krylov",markershape = :pentagon,color="orange",yaxis=:log)
plot!(plot2,range_test,Krylov_error_step,label="",color="orange",yaxis=:log)
scatter!(plot2,range_test,funDiagPP_error_step,label="funDiag++",markershape = :rect,color="brown",yaxis=:log)
plot!(plot2,range_test,funDiagPP_error_step,label="",color="brown",yaxis=:log)
scatter!(plot2,range_test,funNy_error_step,label="funNyström",color="pink",markershape = :utriangle,yaxis=:log)
plot!(plot2,range_test,funNy_error_step,label="",color="pink",yaxis=:log)


plot3 = scatter(range_test,Exact_fA_error_poly,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot3,range_test,Exact_fA_error_poly,label="",color="red",yaxis=:log)
scatter!(plot3,range_test,Chebyshev_error_poly,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot3,range_test,Chebyshev_error_poly,label="",color="green",yaxis=:log)
scatter!(plot3,range_test,Remez_error_poly,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!(plot3,range_test,Remez_error_poly,label="",color="yellow",yaxis=:log)
scatter!(plot3,range_test,Krylov_error_poly,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot3,range_test,Krylov_error_poly,label="",color="orange",yaxis=:log)
scatter!(plot3,range_test,funDiagPP_error_poly,label="",color="brown",markershape = :rect,yaxis=:log)
plot!(plot3,range_test,funDiagPP_error_poly,label="",color="brown",yaxis=:log)
scatter!(plot3,range_test,funNy_error_poly,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!(plot3,range_test,funNy_error_poly,label="",color="pink",yaxis=:log)

plot4 = scatter(range_test,Exact_fA_error_exp,label="", title="Exp", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot4,range_test,Exact_fA_error_exp,label="",color="red",yaxis=:log)
scatter!(plot4,range_test,Chebyshev_error_exp,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot4,range_test,Chebyshev_error_exp,label="",color="green",yaxis=:log)
scatter!(plot4,range_test,Remez_error_exp,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!(plot4,range_test,Remez_error_exp,label="",color="yellow",yaxis=:log)
scatter!(plot4,range_test,Krylov_error_exp,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot4,range_test,Krylov_error_exp,label="",color="orange",yaxis=:log)
scatter!(plot4,range_test,funDiagPP_error_exp,label="",color="brown",markershape = :rect,yaxis=:log)
plot!(plot4,range_test,funDiagPP_error_exp,label="",color="brown",yaxis=:log)
scatter!(plot4,range_test,funNy_error_exp,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!(plot4,range_test,funNy_error_exp,label="",color="pink",yaxis=:log)

plot(plot1, plot2, plot3, plot4, layout = l,size=(900,600))

#savefig("Paperplot_exp.pdf")










# plot5 = scatter(range_test,Exact_fA_error_flat,label="", title="", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",markershape = :pentagon,yaxis=:log,legend=:topright,size=(1250,400),margin=4mm)
# plot!(plot5,range_test,Exact_fA_error_flat,label="",color="red",yaxis=:log)
# scatter!(plot5,range_test,Chebyshev_error_flat,label="",color="green",markershape = :ltriangle,yaxis=:log)
# plot!(plot5,range_test,Chebyshev_error_flat,label="",color="green",yaxis=:log)
# scatter!(plot5,range_test,Remez_error_flat,label="",color="yellow",markershape = :diamond,yaxis=:log)
# plot!(plot5,range_test,Remez_error_flat,label="",color="yellow",yaxis=:log)
# scatter!(plot5,range_test,Krylov_error_flat,label="",color="orange",markershape = :pentagon,yaxis=:log)
# plot!(plot5,range_test,Krylov_error_flat,label="",color="orange",yaxis=:log)





























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
range_test = 20:40:300
n_tries = 8
lim_q = length(range_test)

Chebyshev_error_flat_inv = zeros(lim_q)
Chebyshev_error_flat_inv2 = zeros(lim_q)
Remez_error_flat_inv = zeros(lim_q)
Krylov_error_flat_inv = zeros(lim_q)
Krylov_error_flat_inv2 = zeros(lim_q)
Stocherr_flat_inv = zeros(lim_q)

Chebyshev_error_step_inv = zeros(lim_q)
Chebyshev_error_step_inv2 = zeros(lim_q)
Remez_error_step_inv = zeros(lim_q)
Krylov_error_step_inv = zeros(lim_q)
Krylov_error_step_inv2 = zeros(lim_q)
Stocherr_step_inv = zeros(lim_q)

Chebyshev_error_poly_inv = zeros(lim_q)
Chebyshev_error_poly_inv2 = zeros(lim_q)
Remez_error_poly_inv = zeros(lim_q)
Krylov_error_poly_inv = zeros(lim_q)
Krylov_error_poly_inv2 = zeros(lim_q)
Stocherr_poly_inv = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            ch_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			# ch_flat2=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, 100,int=(1.0,3.0),maxqueries=range_test[i])
			# rem_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
            kr_flat=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
			kr_flat2=EstimateFunctionDiagonal(A_flat,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, 101,maxqueries=range_test[i])
			GH_exact_inv_flat=EstimateDiagonal(A_flat_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_flat_inv[i] += (1/n_tries)*norm(ch_flat-diag_flat_inv)/norm_flat_inv
			# Chebyshev_error_flat_inv2[i] += (1/n_tries)*norm(ch_flat2-diag_flat_inv)/norm_flat_inv
            # Remez_error_flat_inv[i] += (1/n_tries)*norm(rem_flat-diag_flat_inv)/norm_flat_inv
            Krylov_error_flat_inv[i] += (1/n_tries)*norm(kr_flat-diag_flat_inv)/norm_flat_inv
			Krylov_error_flat_inv2[i] += (1/n_tries)*norm(kr_flat2-diag_flat_inv)/norm_flat_inv
            Stocherr_flat_inv[i] += (1/n_tries)*norm(GH_exact_inv_flat-diag_flat_inv)/norm_flat_inv


            ch_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.001,1.0),maxqueries=range_test[i])
			# ch_step2=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, 100,int=(0.001,1.0),maxqueries=range_test[i])
			# rem_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(0.001,1.0),maxqueries=range_test[i])
            kr_step=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
			kr_step2=EstimateFunctionDiagonal(A_step,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, 101,maxqueries=range_test[i])
            GH_exact_inv_step=EstimateDiagonal(A_step_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_step_inv[i] += (1/n_tries)*norm(ch_step-diag_step_inv)/norm_step_inv
			# Chebyshev_error_step_inv2[i] += (1/n_tries)*norm(ch_step2-diag_step_inv)/norm_step_inv
            # Remez_error_step_inv[i] += (1/n_tries)*norm(rem_step-diag_step_inv)/norm_step_inv
            Krylov_error_step_inv[i] += (1/n_tries)*norm(kr_step-diag_step_inv)/norm_step_inv
			Krylov_error_step_inv2[i] += (1/n_tries)*norm(kr_step2-diag_step_inv)/norm_step_inv
            Stocherr_step_inv[i] += (1/n_tries)*norm(GH_exact_inv_step-diag_step_inv)/norm_step_inv


            ch_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.000001,1.0),maxqueries=range_test[i])
			# ch_poly2=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, 100,int=(0.000001,1.0),maxqueries=range_test[i])
			# rem_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :Remez, 100,int=(0.000001,1.0),maxqueries=range_test[i])
            kr_poly=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, deg+1,maxqueries=range_test[i])
			kr_poly2=EstimateFunctionDiagonal(A_poly,finv,finv,:GirardHutchinson,:queries, :Gaussian, :CG, 101,maxqueries=range_test[i])
            GH_exact_inv_poly=EstimateDiagonal(A_poly_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])

            Chebyshev_error_poly_inv[i] += (1/n_tries)*norm(ch_poly-diag_poly_inv)/norm_poly_inv
			# Chebyshev_error_poly_inv2[i] += (1/n_tries)*norm(ch_poly2-diag_poly_inv)/norm_poly_inv
            # Remez_error_poly_inv[i] += (1/n_tries)*norm(rem_poly-diag_poly_inv)/norm_poly_inv
            Krylov_error_poly_inv[i] += (1/n_tries)*norm(kr_poly-diag_poly_inv)/norm_poly_inv
			Krylov_error_poly_inv2[i] += (1/n_tries)*norm(kr_poly2-diag_poly_inv)/norm_poly_inv
            Stocherr_poly_inv[i] += (1/n_tries)*norm(GH_exact_inv_poly-diag_poly_inv)/norm_poly_inv
    end
end


## Plot the results
#plot(1:10,1:10)
l2 = @layout [a b c]

plot11 = scatter(range_test,Stocherr_flat_inv,label=L"\textrm{GH }+f(\textbf{A})", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",markershape = :pentagon,yaxis=:log,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot11,range_test,Stocherr_flat_inv,label="",color="red",yaxis=:log)
scatter!(plot11,range_test,Chebyshev_error_flat_inv,label="funDiag Chebyshev, deg 25",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot11,range_test,Chebyshev_error_flat_inv,label="",color="green",yaxis=:log)
# scatter!(plot11,range_test,Chebyshev_error_flat_inv2,label="funDiag Chebyshev, deg 100",color="yellow",markershape = :diamond,yaxis=:log)
# plot!(plot11,range_test,Chebyshev_error_flat_inv2,label="",color="yellow",yaxis=:log)
scatter!(plot11,range_test,Krylov_error_flat_inv,label="funDiag CG, 26 steps",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot11,range_test,Krylov_error_flat_inv,label="",color="orange",yaxis=:log)
scatter!(plot11,range_test,Krylov_error_flat_inv,label="funDiag CG, 101 steps",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot11,range_test,Krylov_error_flat_inv,label="",color="brown",yaxis=:log)


plot12 = scatter(range_test,Stocherr_step_inv,label="", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :circle,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot12,range_test,Stocherr_step_inv,label="",color="red",yaxis=:log)
scatter!(plot12,range_test,Chebyshev_error_step_inv,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot12,range_test,Chebyshev_error_step_inv,label="",color="green",yaxis=:log)
# scatter!(plot12,range_test,Chebyshev_error_step_inv2,label="funDiag ",color="blue",markershape = :ltriangle,yaxis=:log)
# plot!(plot12,range_test,Chebyshev_error_step_inv2,label="",color="blue",yaxis=:log)
scatter!(plot12,range_test,Krylov_error_step_inv,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot12,range_test,Krylov_error_step_inv,label="",color="orange",yaxis=:log)
scatter!(plot12,range_test,Krylov_error_step_inv2,label="",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot12,range_test,Krylov_error_step_inv2,label="",color="brown",yaxis=:log)


plot13 = scatter(range_test,Stocherr_poly_inv,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot13,range_test,Stocherr_poly_inv,label="",color="red",yaxis=:log)
scatter!(plot13,range_test,Chebyshev_error_poly_inv,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot13,range_test,Chebyshev_error_poly_inv,label="",color="green",yaxis=:log)
# scatter!(plot13,range_test,Chebyshev_error_poly_inv2,label="",color="blue",markershape = :ltriangle,yaxis=:log)
# plot!(plot13,range_test,Chebyshev_error_poly_inv2,label="",color="blue",yaxis=:log)
scatter!(plot13,range_test,Krylov_error_poly_inv,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot13,range_test,Krylov_error_poly_inv,label="",color="orange",yaxis=:log)
scatter!(plot13,range_test,Krylov_error_poly_inv2,label="",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot13,range_test,Krylov_error_poly_inv2,label="",color="brown",yaxis=:log)



plot(plot11, plot12, plot13, layout = l2,size=(900,300))

#savefig("Paperplot_inv.pdf")





















##
############################
# Logarithm
############################
flog = x->log(x+1)
logmat = A->log(A+Matrix(I,size(A)))

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

## Perform experiments
n_tries = 1
range_test = 20:20:300
lim_q = length(range_test)
deg = 25
q_test=6

Exact_fA_error_flat_log = zeros(lim_q)
Exact_fA_error_poly_log = zeros(lim_q)
Exact_fA_error_exp_log = zeros(lim_q)
Exact_fA_error_step_log = zeros(lim_q)

Chebyshev_error_flat_log = zeros(lim_q)
Chebyshev_error_poly_log = zeros(lim_q)
Chebyshev_error_exp_log = zeros(lim_q)
Chebyshev_error_step_log = zeros(lim_q)

Remez_error_flat_log = zeros(lim_q)
Remez_error_poly_log = zeros(lim_q)
Remez_error_exp_log = zeros(lim_q)
Remez_error_step_log = zeros(lim_q)

Krylov_error_flat_log = zeros(lim_q)
Krylov_error_poly_log = zeros(lim_q)
Krylov_error_exp_log = zeros(lim_q)
Krylov_error_step_log = zeros(lim_q)

funDiagPP_error_flat_log = zeros(lim_q)
funDiagPP_error_poly_log = zeros(lim_q)
funDiagPP_error_exp_log = zeros(lim_q)
funDiagPP_error_step_log = zeros(lim_q)

funNy_error_flat_log = zeros(lim_q)
funNy_error_poly_log = zeros(lim_q)
funNy_error_exp_log = zeros(lim_q)
funNy_error_step_log = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
			GH_exact_exp_flat=EstimateDiagonal(A_flat_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_flat=EstimateFunctionDiagonal(A_flat,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_flat=EstimateFunctionDiagonal(A_flat,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_flat=GirardHutchinsonFunctionDiagonalKrylovCustom(A_flat,range_test[i],logmat,deg+1)
			# Add try-catck block to handle LAPACK exceptions
			fundiagpp_flat=zeros(n)
			funNy_flat=zeros(n)
			try
				fundiagpp_flat=KrylovfunDiagPP(A_flat, logmat,flog,range_test[i],deg+1,q_test)
				funNy_flat=funNyströmDiag(A_flat, logmat,flog,range_test[i],deg+1,q_test)
			catch
				fundiagpp_flat=KrylovfunDiagPP(A_flat, logmat,flog,range_test[i],deg+1,q_test)
				funNy_flat=funNyströmDiag(A_flat, logmat,flog,range_test[i],deg+1,q_test)
			end


            Exact_fA_error_flat_log[i] += (1/n_tries)*norm(GH_exact_exp_flat-diag_flat_log)/norm_flat_log
            Chebyshev_error_flat_log[i] += (1/n_tries)*norm(ch_flat-diag_flat_log)/norm_flat_log
            Remez_error_flat_log[i] += (1/n_tries)*norm(rem_flat-diag_flat_log)/norm_flat_log
            Krylov_error_flat_log[i] += (1/n_tries)*norm(kr_flat-diag_flat_log)/norm_flat_log
            funDiagPP_error_flat_log[i] += (1/n_tries)*norm(fundiagpp_flat-diag_flat_log)/norm_flat_log
			funNy_error_flat_log[i] += (1/n_tries)*norm(funNy_flat-diag_flat_log)/norm_flat_log

			GH_exact_exp_step=EstimateDiagonal(A_step_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_step=EstimateFunctionDiagonal(A_step,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_step=EstimateFunctionDiagonal(A_step,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_step=GirardHutchinsonFunctionDiagonalKrylovCustom(A_step,range_test[i],logmat,deg+1)
			fundiagpp_step=zeros(n)
			funNy_step=zeros(n)
			try
				fundiagpp_step=KrylovfunDiagPP(A_step, logmat,flog_norm,range_test[i],deg+1,q_test)
				funNy_step=funNyströmDiag(A_step, logmat,flog_norm,range_test[i],deg+1,q_test)
			catch
				fundiagpp_step=KrylovfunDiagPP(A_step, logmat,flog,range_test[i],deg+1,q_test)
				funNy_step=funNyströmDiag(A_step, logmat,flog,range_test[i],deg+1,q_test)
			end

			Exact_fA_error_step_log[i] += (1/n_tries)*norm(GH_exact_exp_step-diag_step_log)/norm_step_log
			Chebyshev_error_step_log[i] += (1/n_tries)*norm(ch_step-diag_step_log)/norm_step_log
			Remez_error_step_log[i] += (1/n_tries)*norm(rem_step-diag_step_log)/norm_step_log
			Krylov_error_step_log[i] += (1/n_tries)*norm(kr_step-diag_step_log)/norm_step_log
			funDiagPP_error_step_log[i] += (1/n_tries)*norm(fundiagpp_step-diag_step_log)/norm_step_log
			funNy_error_step_log[i] += (1/n_tries)*norm(funNy_step-diag_step_log)/norm_step_log

			GH_exact_exp_poly=EstimateDiagonal(A_poly_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_poly=EstimateFunctionDiagonal(A_poly,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_poly=EstimateFunctionDiagonal(A_poly,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_poly=GirardHutchinsonFunctionDiagonalKrylovCustom(A_poly,range_test[i],logmat,deg+1)
			fundiagpp_poly=zeros(n)
			funNy_poly=zeros(n)
			try
				fundiagpp_poly=KrylovfunDiagPP(A_poly, logmat,flog,range_test[i],deg+1,q_test)
				funNy_poly=funNyströmDiag(A_poly, logmat,flog,range_test[i],deg+1,q_test)
			catch
				fundiagpp_poly=KrylovfunDiagPP(A_poly, logmat,flog,range_test[i],deg+1,q_test)
				funNy_poly=funNyströmDiag(A_poly, logmat,flog,range_test[i],deg+1,q_test)
			end


			Exact_fA_error_poly_log[i] += (1/n_tries)*norm(GH_exact_exp_poly-diag_poly_log)/norm_poly_log
			Chebyshev_error_poly_log[i] += (1/n_tries)*norm(ch_poly-diag_poly_log)/norm_poly_log
			Remez_error_poly_log[i] += (1/n_tries)*norm(rem_poly-diag_poly_log)/norm_poly_log
			Krylov_error_poly_log[i] += (1/n_tries)*norm(kr_poly-diag_poly_log)/norm_poly_log
			funDiagPP_error_poly_log[i] += (1/n_tries)*norm(diag_poly_log-diag_poly_exp)/norm_poly_exp
			funNy_error_poly_log[i] += (1/n_tries)*norm(funNy_poly-diag_poly_log)/norm_poly_log

			GH_exact_exp_exp=EstimateDiagonal(A_exp_log,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			ch_exp=EstimateFunctionDiagonal(A_exp,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			rem_exp=EstimateFunctionDiagonal(A_exp,flog,flog,:GirardHutchinson,:queries, :Gaussian, :Remez, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_exp=GirardHutchinsonFunctionDiagonalKrylovCustom(A_exp,range_test[i],logmat,deg+1)
			fundiagpp_exp=zeros(n)
			funNy_exp=zeros(n)
			try
				fundiagpp_exp=KrylovfunDiagPP(A_exp, logmat,flog,range_test[i],deg+1,q_test)
				funNy_exp=funNyströmDiag(A_exp, logmat,flog,range_test[i],deg+1,q_test)
			catch
				fundiagpp_exp=KrylovfunDiagPP(A_exp, logmat,flog,range_test[i],deg+1,q_test)
				funNy_exp=funNyströmDiag(A_exp, logmat,flog,range_test[i],deg+1,q_test)
			end


            Exact_fA_error_exp_log[i] += (1/n_tries)*norm(GH_exact_exp_exp-diag_exp_log)/norm_exp_log
            Chebyshev_error_exp_log[i] += (1/n_tries)*norm(ch_exp-diag_exp_log)/norm_exp_log
            Remez_error_exp_log[i] += (1/n_tries)*norm(rem_exp-diag_exp_log)/norm_exp_log
            Krylov_error_exp_log[i] += (1/n_tries)*norm(kr_exp-diag_exp_log)/norm_exp_log
            funDiagPP_error_exp_log[i] += (1/n_tries)*norm(fundiagpp_exp-diag_exp_log)/norm_exp_log
			funNy_error_exp_log[i] += (1/n_tries)*norm(funNy_exp-diag_exp_log)/norm_exp_log
    end
end

##
l = @layout [a b;c d]

plot11 = scatter(range_test,Exact_fA_error_flat_log,label="", title="flat_log", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",markershape = :pentagon,yaxis=:log,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot11,range_test,Exact_fA_error_flat_log,label="",color="red",yaxis=:log)
scatter!(plot11,range_test,Chebyshev_error_flat_log,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot11,range_test,Chebyshev_error_flat_log,label="",color="green",yaxis=:log)
scatter!(plot11,range_test,Remez_error_flat_log,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!(plot11,range_test,Remez_error_flat_log,label="",color="yellow",yaxis=:log)
scatter!(plot11,range_test,Krylov_error_flat_log,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot11,range_test,Krylov_error_flat_log,label="",color="orange",yaxis=:log)
scatter!(plot11,range_test,funDiagPP_error_flat_log,label="",color="brown",markershape = :rect,yaxis=:log)
plot!(plot11,range_test,funDiagPP_error_flat_log,label="",color="brown",yaxis=:log)
scatter!(plot11,range_test,funNy_error_flat_log,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!(plot11,range_test,funNy_error_flat_log,label="",color="pink",yaxis=:log)


 plot22 = scatter(range_test,Exact_fA_error_step_log,label=L"\textrm{GH }+f(\textbf{A})", title="Step_log", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :circle,legend=:topright,size=(1250,400),margin=4mm)
plot!( plot22,range_test,Exact_fA_error_step_log,label="",color="red",yaxis=:log)
scatter!( plot22,range_test,Chebyshev_error_step_log,label="funDiag Chebyshev",color="green",markershape = :ltriangle,yaxis=:log)
plot!( plot22,range_test,Chebyshev_error_step_log,label="",color="green",yaxis=:log)
scatter!( plot22,range_test,Remez_error_step_log,label="funDiag Remez",color="yellow",markershape = :diamond,yaxis=:log)
plot!( plot22,range_test,Remez_error_step_log,label="",color="yellow",yaxis=:log)
scatter!( plot22,range_test,Krylov_error_step_log,label="funDiag Krylov",markershape = :pentagon,color="orange",yaxis=:log)
plot!( plot22,range_test,Krylov_error_step_log,label="",color="orange",yaxis=:log)
scatter!( plot22,range_test,funDiagPP_error_step_log,label="funDiag++",markershape = :rect,color="brown",yaxis=:log)
plot!( plot22,range_test,funDiagPP_error_step_log,label="",color="brown",yaxis=:log)
scatter!( plot22,range_test,funNy_error_step_log,label="funNyström",color="pink",markershape = :utriangle,yaxis=:log)
plot!( plot22,range_test,funNy_error_step_log,label="",color="pink",yaxis=:log)


 plot23 = scatter(range_test,Exact_fA_error_poly_log,label="", title="Poly_log", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!( plot23,range_test,Exact_fA_error_poly_log,label="",color="red",yaxis=:log)
scatter!( plot23,range_test,Chebyshev_error_poly_log,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!( plot23,range_test,Chebyshev_error_poly_log,label="",color="green",yaxis=:log)
scatter!( plot23,range_test,Remez_error_poly_log,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!( plot23,range_test,Remez_error_poly_log,label="",color="yellow",yaxis=:log)
scatter!( plot23,range_test,Krylov_error_poly_log,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!( plot23,range_test,Krylov_error_poly_log,label="",color="orange",yaxis=:log)
scatter!( plot23,range_test,funDiagPP_error_poly_log,label="",color="brown",markershape = :rect,yaxis=:log)
plot!( plot23,range_test,funDiagPP_error_poly_log,label="",color="brown",yaxis=:log)
scatter!( plot23,range_test,funNy_error_poly_log,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!( plot23,range_test,funNy_error_poly_log,label="",color="pink",yaxis=:log)

 plot24 = scatter(range_test,Exact_fA_error_exp_log,label="", title="Exp_log", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!( plot24,range_test,Exact_fA_error_exp_log,label="",color="red",yaxis=:log)
scatter!( plot24,range_test,Chebyshev_error_exp_log,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!( plot24,range_test,Chebyshev_error_exp_log,label="",color="green",yaxis=:log)
scatter!( plot24,range_test,Remez_error_exp_log,label="",color="yellow",markershape = :diamond,yaxis=:log)
plot!( plot24,range_test,Remez_error_exp_log,label="",color="yellow",yaxis=:log)
scatter!( plot24,range_test,Krylov_error_exp_log,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!( plot24,range_test,Krylov_error_exp_log,label="",color="orange",yaxis=:log)
scatter!( plot24,range_test,funDiagPP_error_exp_log,label="",color="brown",markershape = :rect,yaxis=:log)
plot!( plot24,range_test,funDiagPP_error_exp_log,label="",color="brown",yaxis=:log)
scatter!( plot24,range_test,funNy_error_exp_log,label="",color="pink",markershape = :utriangle,yaxis=:log)
plot!( plot24,range_test,funNy_error_exp_log,label="",color="pink",yaxis=:log)

plot(plot1,  plot22,  plot23,  plot24, layout = l,size=(900,600))

















##
############################
# Double Inverse
############################
A_flat_double_inv  = Q*diagm(inv.(flat_ev.^2))*Q'
A_poly_double_inv  = Q*diagm(inv.(poly_ev.^2))*Q'
A_exp_double_inv  = Q*diagm(inv.(exp_ev.^2))*Q'
A_step_double_inv  = Q*diagm(inv.(step_ev.^2))*Q'

diag_flat_double_inv = diag(A_flat_double_inv)
diag_poly_double_inv = diag(A_poly_double_inv)
diag_exp_double_inv = diag(A_exp_double_inv)
diag_step_double_inv = diag(A_step_double_inv)

norm_flat_double_inv = norm(diag_flat_double_inv)
norm_poly_double_inv = norm(diag_poly_double_inv)
norm_exp_double_inv = norm(diag_exp_double_inv)
norm_step_double_inv = norm(diag_step_double_inv);


## Custom functions

function GirardHutchinsonDiagonalDoubleInverseCG(A,s,c_it,dist=:Gaussian,normal=true,testmatrix=nothing)
	    # A: input matrix, s: number of samples, c_it: number of iteration for conjugate gradient, dist: distribution for random vectors, normalization: select if normalization should be performed or not

	    # get sizes of the matrix
	    (m,n)=size(A)
	    if n!=m
	        print("Matrix must be square")
	    end
	    # Sample random test matrix
	    if dist==:Rademacher
	        d = Bernoulli(.5)
	        Om=2*rand(d,n,s).-1
	    elseif dist==:Gaussian
	        Om=randn(n,s)
	    elseif dist==:custom
	        Om=testmatrix
	    else
	        print("Unknown distribution")
	    end
	    # Perform action of A^-1
	    Action=zeros(n,s)
	    for i=1:s
			intermediate_result = cg(A,float.(Om[:,i]),maxiter=c_it)
	        Action[:,i] = cg(A,float.(vec(intermediate_result)),maxiter=c_it).*Om[:,i]
	    end

	    #Calculate normalization
	    if normal==true
	        V = Om.*Om
	        normalization =  sum(V,dims=2)
	    else
	        normalization = s*ones(n)
	    end
	    # Calculate the final result
	    result = sum(Action,dims=2)./normalization

	    return vec(result)
	end



## Simple plot
fdouble_inv = x->inv(x.^2)
deg = 25

cgdo_res = GirardHutchinsonDiagonalDoubleInverseCG(A_step,200,25)
chebdo_res = EstimateFunctionDiagonal(A_step,finv_double,finv_double,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.001,0.1),maxqueries=200)
worng_res=EstimateDiagonal(A_step,:GirardHutchinson,:queries,:Gaussian,maxqueries=1).^(-2)
#cg(A_flat,randn(n),maxiter=10)


norm(diag_step_double_inv-cgdo_res)/norm_step_double_inv
norm(diag_flat_double_inv-chebdo_res)/norm_flat_double_inv
norm(diag_flat_double_inv-worng_res)/norm_flat_double_inv
##
fdouble_inv = x->inv(x.^2)
deg = 25
range_test = 20:40:300
n_tries = 10
lim_q = length(range_test)

Chebyshev_error_flat_double_inv = zeros(lim_q)
Change_error_flat_double_inv = zeros(lim_q)
Krylov_error_flat_double_inv = zeros(lim_q)
Krylov_error_flat_double_inv2 = zeros(lim_q)
Stocherr_flat_double_inv = zeros(lim_q)

Chebyshev_error_step_double_inv = zeros(lim_q)
Change_error_step_double_inv = zeros(lim_q)
Krylov_error_step_double_inv = zeros(lim_q)
Krylov_error_step_double_inv2 = zeros(lim_q)
Stocherr_step_double_inv = zeros(lim_q)

Chebyshev_error_poly_double_inv = zeros(lim_q)
Change_error_poly_double_inv = zeros(lim_q)
Krylov_error_poly_double_inv = zeros(lim_q)
Krylov_error_poly_double_inv2 = zeros(lim_q)
Stocherr_poly_double_inv = zeros(lim_q)


for i=1:lim_q
    for j=1:n_tries
            ch_flat=EstimateFunctionDiagonal(A_flat,fdouble_inv,fdouble_inv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(1.0,3.0),maxqueries=range_test[i])
			kr_flat=GirardHutchinsonDiagonalDoubleInverseCG(A_flat,range_test[i],deg+1)
			kr_flat2=GirardHutchinsonDiagonalDoubleInverseCG(A_flat,range_test[i],101)
			GH_exact_double_inv_flat=EstimateDiagonal(A_flat_double_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			Change_double_inv_flat=EstimateDiagonal(A_flat,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i]).^(-2)

            Chebyshev_error_flat_double_inv[i] += (1/n_tries)*norm(ch_flat-diag_flat_double_inv)/norm_flat_double_inv
			Krylov_error_flat_double_inv[i] += (1/n_tries)*norm(kr_flat-diag_flat_double_inv)/norm_flat_double_inv
			Krylov_error_flat_double_inv2[i] += (1/n_tries)*norm(kr_flat2-diag_flat_double_inv)/norm_flat_double_inv
            Stocherr_flat_double_inv[i] += (1/n_tries)*norm(GH_exact_double_inv_flat-diag_flat_double_inv)/norm_flat_double_inv
			Change_error_flat_double_inv[i] += (1/n_tries)*norm(Change_double_inv_flat-diag_flat_double_inv)/norm_flat_double_inv


            ch_step=EstimateFunctionDiagonal(A_step,fdouble_inv,fdouble_inv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.001,1.0),maxqueries=range_test[i])
			kr_step=GirardHutchinsonDiagonalDoubleInverseCG(A_step,range_test[i],deg+1)
			kr_step2= GirardHutchinsonDiagonalDoubleInverseCG(A_step,range_test[i],101)
            GH_exact_double_inv_step=EstimateDiagonal(A_step_double_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			Change_double_inv_step=EstimateDiagonal(A_step,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i]).^(-2)

            Chebyshev_error_step_double_inv[i] += (1/n_tries)*norm(ch_step-diag_step_double_inv)/norm_step_double_inv
			Krylov_error_step_double_inv[i] += (1/n_tries)*norm(kr_step-diag_step_double_inv)/norm_step_double_inv
			Krylov_error_step_double_inv2[i] += (1/n_tries)*norm(kr_step2-diag_step_double_inv)/norm_step_double_inv
            Stocherr_step_double_inv[i] += (1/n_tries)*norm(GH_exact_double_inv_step-diag_step_double_inv)/norm_step_double_inv
			Change_error_step_double_inv[i] += (1/n_tries)*norm(Change_double_inv_step-diag_step_double_inv)/norm_step_double_inv



            ch_poly=EstimateFunctionDiagonal(A_poly,fdouble_inv,fdouble_inv,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, deg,int=(0.000001,1.0),maxqueries=range_test[i])
			kr_poly=GirardHutchinsonDiagonalDoubleInverseCG(A_poly,range_test[i],deg+1)
			kr_poly2=GirardHutchinsonDiagonalDoubleInverseCG(A_poly,range_test[i],101)
			GH_exact_double_inv_poly=EstimateDiagonal(A_poly_double_inv,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i])
			Change_double_inv_poly=EstimateDiagonal(A_poly,:GirardHutchinson,:queries,:Gaussian,maxqueries=range_test[i]).^(-2)

			Chebyshev_error_poly_double_inv[i] += (1/n_tries)*norm(ch_poly-diag_poly_double_inv)/norm_poly_double_inv
			Krylov_error_poly_double_inv[i] += (1/n_tries)*norm(kr_poly-diag_poly_double_inv)/norm_poly_double_inv
			Krylov_error_poly_double_inv2[i] += (1/n_tries)*norm(kr_poly2-diag_poly_double_inv)/norm_poly_double_inv
            Stocherr_poly_double_inv[i] += (1/n_tries)*norm(GH_exact_double_inv_poly-diag_poly_double_inv)/norm_poly_double_inv
			Change_error_poly_double_inv[i] += (1/n_tries)*norm(Change_double_inv_poly-diag_poly_double_inv)/norm_poly_double_inv
	end
end

## Plot the results
#plot(1:10,1:10)
l2 = @layout [a b c]

plot51 = scatter(range_test,Stocherr_flat_double_inv,label=L"\textrm{GH }+f(\textbf{A})", title="Flat", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",color="red",markershape = :pentagon,yaxis=:log,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot51,range_test,Stocherr_flat_double_inv,label="",color="red",yaxis=:log)
scatter!(plot51,range_test,Chebyshev_error_flat_double_inv,label="funDiag Chebyshev, deg 25",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot51,range_test,Chebyshev_error_flat_double_inv,label="",color="green",yaxis=:log)
scatter!(plot51,range_test,Krylov_error_flat_double_inv,label="funDiag CG, 26 steps",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot51,range_test,Krylov_error_flat_double_inv,label="",color="orange",yaxis=:log)
scatter!(plot51,range_test,Krylov_error_flat_double_inv,label="funDiag CG, 101 steps",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot51,range_test,Krylov_error_flat_double_inv,label="",color="brown",yaxis=:log)
scatter!(plot51,range_test,Change_error_flat_double_inv,label=L"D^s(\textbf(A))^{-2}",color="black",markershape = :cross,yaxis=:log)
plot!(plot51,range_test,Change_error_flat_double_inv,label="",color="black",yaxis=:log)


plot52 = scatter(range_test,Stocherr_step_double_inv,label="", title="Step", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :circle,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot52,range_test,Stocherr_step_double_inv,label="",color="red",yaxis=:log)
scatter!(plot52,range_test,Chebyshev_error_step_double_inv,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot52,range_test,Chebyshev_error_step_double_inv,label="",color="green",yaxis=:log)
scatter!(plot52,range_test,Krylov_error_step_double_inv,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot52,range_test,Krylov_error_step_double_inv,label="",color="orange",yaxis=:log)
scatter!(plot52,range_test,Krylov_error_step_double_inv2,label="",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot52,range_test,Krylov_error_step_double_inv2,label="",color="brown",yaxis=:log)
scatter!(plot52,range_test,Change_error_step_double_inv,label=L"D^s(\textbf(A))^{-2}",color="black",markershape = :cross,yaxis=:log)
plot!(plot52,range_test,Change_error_step_double_inv,label="",color="black",yaxis=:log)

plot53 = scatter(range_test,Stocherr_poly_double_inv,label="", title="Poly", ylabel=L"\textrm{Relative }\mathcal{L}^2\textrm{ error}", xlabel="Number of test vectors",yaxis=:log,color="red",markershape = :pentagon,legend=:topright,size=(1250,400),margin=4mm)
plot!(plot53,range_test,Stocherr_poly_double_inv,label="",color="red",yaxis=:log)
scatter!(plot53,range_test,Chebyshev_error_poly_double_inv,label="",color="green",markershape = :ltriangle,yaxis=:log)
plot!(plot53,range_test,Chebyshev_error_poly_double_inv,label="",color="green",yaxis=:log)
scatter!(plot53,range_test,Krylov_error_poly_double_inv,label="",color="orange",markershape = :pentagon,yaxis=:log)
plot!(plot53,range_test,Krylov_error_poly_double_inv,label="",color="orange",yaxis=:log)
scatter!(plot53,range_test,Krylov_error_poly_double_inv2,label="",color="brown",markershape = :pentagon,yaxis=:log)
plot!(plot53,range_test,Krylov_error_poly_double_inv2,label="",color="brown",yaxis=:log)
scatter!(plot53,range_test,Change_error_poly_double_inv,label=L"D^s(\textbf(A))^{-2}",color="black",markershape = :cross,yaxis=:log)
plot!(plot53,range_test,Change_error_poly_double_inv,label="",color="black",yaxis=:log)


plot(plot51, plot52, plot53, layout = l2,size=(900,300))
