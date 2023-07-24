## Examples
The usage of the three function exported by the package is very simple. First, let's create an example matrix whose diagonal is easy to estimate.
```@example
n=20
Om = randn(n,n)
Q,R = qr(Om)
A_temp = Q*diagm(vec(1:n).^(-4))*Q'
A_temp = A_temp*A_temp'
A=A_temp
A = 10^(-8)*A_temp
for i=1:n
    A[i,i]=10^14*A[i,i]
end
true_diag = diag(A)
```
Standard diagonal estimation when using a fixed number of samples is carried out as follows.
```@example
# Girard-Hutchinson
EstimateDiagonal(A,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=100)≈true_diag
# Diag++
EstimateDiagonal(A,:DiagPP, :queries, :Rademacher, true,maxqueries=100)≈true_diag
# NysDiag++
EstimateDiagonal(A,:NysDiagPP, :queries, :Rademacher, true,maxqueries=100)≈true_diag
# XDiag
EstimateDiagonal(A,:XDiag, :queries, :Rademacher, true,maxqueries=100)≈true_diag
```
When using the doubling strategy the third keyword has to be altered.
```@example
# Girard-Hutchinson with doubling
EstimateMoMDiagonal(A,:GirardHutchinson, :doubling, :Rademacher,true,queries_start=100,eps=0.2)
# Diag++ with doubling
EstimateMoMDiagonal(A,:DiagPP, :doubling, :Rademacher,true,queries_start=100,eps=0.2)
```
Using the median of means versions with 100 subgroups of size 1000 is carried out by the `EstimateMoMDiagonal` function.
```@example
# Girard-Hutchinson
EstimateMoMDiagonal(A,:GirardHutchinson, :queries, :Rademacher, 100,900,true,maxqueries=100)
# Diag++
EstimateMoMDiagonal(A,:DiagPP, :queries, :Rademacher, 100,900,true,maxqueries=100)
# NysDiag++
EstimateMoMDiagonal(A,:NysDiagPP, :queries, :Rademacher, 100,900,true,maxqueries=100)
# XDiag
EstimateMoMDiagonal(A,:XDiag, :queries, :Rademacher, 100,900,true,maxqueries=100)
```
ADiag++ can be used by setting the second keyword to `:DiagPP` and the third keyword to `:adaptive`. The example shown next creates an (0.1,0.001) estimator.
```@example
# Girard-Hutchinson
RandomizedDiagonalEstimation.EstimateDiagonal(A,:DiagPP, :adaptive, :Rademacher, true,epsilon=0.1, delta=0.001)
```
For testing the methods available matrix functions, we create another test matrix.
```@example
fexp = x -> exp(x)
Om = randn(n,n)
Q,R = qr(Om)
A_2 = Q*diagm(vec(1:n).^(-4))*Q'
A_2_exp = Q*diagm(exp.(vec(1:n).^(-4)))*Q'
diag_exp = diag(A_2_exp)
```
There are three different types of approximators for general matrix functions which can be used for the estimation of the diagonal in conjunction with the Girard-Hutchinson estimator. They all use the same syntax and are specified by the sixth keyword.
```@example
# Remez Polynomials
EstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, 25,int=(0.0,1.0),maxqueries=10000)
# Remez Interpolants
EstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev,25,int=(0.0,1.0),maxqueries=10000)
# Arnoldi Approximation
EstimateFunctionDiagonal(A_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov,18,int=(0.0,1.0),maxqueries=10000)
```
For the diagonal of the inverse we provide an additional estimator based on the conjugate gradients method.
```@example
# Reference solution
A_2_inv = Q*diagm((vec(1:n).^(2)))*Q'
diag_inv = diag(A_temp_2_inv)
# Estimator
finv = x -> x^(-1)
EstimateFunctionDiagonal(A_2,finv,:GirardHutchinson,:queries, :Gaussian, :CG, 25,int=(0.0,1.0),maxqueries=10000)
```
