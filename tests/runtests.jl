using Test
using Random
using LinearAlgebra
using RandomizedDiagonalEstimation

## Define Test Matrices
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
## Test standard functions

@testset "RandomizedDiagonalEstimation.jl" begin
    @test RandomizedDiagonalEstimation.EstimateDiagonal(A,:GirardHutchinson, :queries, :Rademacher, true,maxqueries=1000)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateDiagonal(A,:DiagPP, :queries, :Rademacher, true,maxqueries=1000)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateDiagonal(A,:NysDiagPP, :queries, :Rademacher, true,maxqueries=40)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateDiagonal(A,:XDiag, :queries, :Rademacher, true,maxqueries=1000)≈true_diag
end

## Test MoM

@testset "RandomizedDiagonalEstimation.jl" begin
    @test RandomizedDiagonalEstimation.EstimateMoMDiagonal(A,:GirardHutchinson, :queries, :Rademacher, 1000,900,true,maxqueries=1000)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateMoMDiagonal(A,:DiagPP, :queries, :Rademacher, 1000,250,true,maxqueries=1000)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateMoMDiagonal(A,:NysDiagPP, :queries, :Rademacher, 1000,19,true,maxqueries=40)≈true_diag
    @test RandomizedDiagonalEstimation.EstimateMoMDiagonal(A,:XDiag, :queries, :Rademacher, 1000,250,true,maxqueries=1000)≈true_diag
end

## Test Adaptive
eps_test=100*norm(true_diag)
@testset "RandomizedDiagonalEstimation.jl" begin
    @test norm(RandomizedDiagonalEstimation.EstimateDiagonal(A,:DiagPP, :adaptive, :Rademacher, true,epsilon=0.1eps_test, delta=0.001)-true_diag)<eps_test
end

## Test Functions
fexp = x -> exp(x)
finv = x -> x^(-1)
Om = randn(n,n)
Q,R = qr(Om)
A_temp_2 = Q*diagm(vec(1:n).^(-4))*Q'
A_temp_2_exp = Q*diagm(exp.(vec(1:n).^(-4)))*Q'
diag_exp = diag(A_temp_2_exp)
diag_exp

@testset "RandomizedDiagonalEstimation.jl" begin
    @test norm(RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_temp_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Remez, 25,int=(0.0,1.0),maxqueries=10000)-diag_exp)/norm(diag_exp)<1
    @test norm(RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_temp_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Chebyshev, 25,int=(0.0,1.0),maxqueries=10000)-diag_exp)/norm(diag_exp)<1
    @test norm(RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_temp_2,fexp,:GirardHutchinson,:queries, :Gaussian, :Krylov, 18,int=(0.0,1.0),maxqueries=10000)-diag_exp)/norm(diag_exp)<1
end

A_temp_2 = Q*diagm(vec(1:n).^(-2))*Q'
A_temp_2_inv = Q*diagm((vec(1:n).^(2)))*Q'
diag_inv = diag(A_temp_2_inv)
diag_inv

@testset "RandomizedDiagonalEstimation.jl" begin
    @test norm(RandomizedDiagonalEstimation.EstimateFunctionDiagonal(A_temp_2,fexp,:GirardHutchinson,:queries, :Gaussian, :CG, 25,int=(0.0,1.0),maxqueries=10000)-diag_inv)/norm(diag_inv)<1
end
