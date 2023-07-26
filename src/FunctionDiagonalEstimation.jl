function GirardHutchinsonFunctionDiagonalPolyApprox(A,s,f,p,int,dist=:Gaussian,normal=true,testmatrix=nothing)
    # A: input matrix, s: number of samples, p: degree of the polynomial approximation, dist: distribution for random vectors, normalization: select if normalization should be performed or not

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    Om=zeros(n,s)
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
    # orthogonalize test matrix
    # Q,R=qr(Om)
    # Om=Matrix(Q)
    # Approximate the function f polynomially
    S = Chebyshev(int[1]..int[2])
    x = points(S, p)
    y = f.(x)
    f_fit = Polynomials.fit(x, y);

    # Perform action based on polynomial
    Action=zeros(n,s)
    for j=1:s
        x_result=zeros(n)
        potenz = Om[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:p
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
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

function GirardHutchinsonFunctionDiagonalKrylov(A,s,f,m_Kr,dist=:Gaussian,normal=true,orthogonalize=false,testmatrix=nothing)
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
        Action[:,j]=KrylovMatfunc(A,Om[:,j],f,m_Kr)
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

function GirardHutchinsonDiagonalInverseCG(A,s,c_it,dist=:Gaussian,normal=true,testmatrix=nothing)
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
        Action[:,i] = cg(A,float.(Om[:,i]),itmax=c_it)[1].*Om[:,i]
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

function GirardHutchinsonFunctionDiagonalRemez(A,s,f,p,int,dist=:Gaussian,normal=true,testmatrix=nothing)
    # A: input matrix, s: number of samples, p: degree of the polynomial approximation, dist: distribution for random vectors, normalization: select if normalization should be performed or not

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
    # orthogonalize test matrix
    # Q,R=qr(Om)
    # Om=Matrix(Q)
    # Approximate the function f polynomially
    f_fit = RemezPoly(f,int,p)

    # Perform action based on polynomial
    Action=zeros(n,s)
    for j=1:s
        x_result=zeros(n)
        potenz = Om[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:p
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
    end
    #print(Action - Ainv*Om)
    Action=Action.*Om

    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        V = ones(size(Om))
        normalization = s*ones(n)
    end
    # Calculate the final result
    result = sum(Action,dims=2)./normalization

    return vec(result)
end

function funNyströmDiag(A,f,k,q=4)
    U,S=nystrom(A,k,q)
    A_nys_exp=U*diagm(f.(S))*U'
    return diag(A_nys_exp)
end

function funDiagPP_Krylov(A, fAfun,fscalar,s_in,deg,dist,normal=true,q=4)

    r = Int(round(s_in/3))
    m = s_in-2*r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
    U, S = nystrom(A, r, q)
    #t1 = sum(diagm(fscalar.(S)))
    N = U*diagm(fscalar.(S))*U'
    diag1=diag(N)

    # Stochastic trace estimation phase
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,m).-1
    elseif dist==:Gaussian
        Om=randn(n,m)
    elseif dist==:custom
        Om=testmatrix
    else
        print("Unknown distribution")
    end
    Action=zeros(n,m)
    for j=1:m
        # Compute Krylov subspace
        Action[:,j]=KrylovMatfunc(A,Om[:,j],fAfun,deg)
    end
    Action=Action.*Om-(N*Om).*Om

    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    diag2 = sum(Action,dims=2)./normalization

    # Return result
    return diag1+diag2
end

function funDiagPP_Remez(A, fAfun,fscalar,s_in,deg,int,dist,normal=true,q=4)
    r = Int(round(s_in/3))
    m = s_in-2*r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
    U, S = nystrom(A, r, q)
    #t1 = sum(diagm(fscalar.(S)))
    N = U*diagm(fscalar.(S))*U'
    diag1=diag(N)

    # Stochastic trace estimation phase
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,m).-1
    elseif dist==:Gaussian
        Om=randn(n,m)
    elseif dist==:custom
        Om=testmatrix
    else
        print("Unknown distribution")
    end
    # fit Remez polynomial
    f_fit = RemezPoly(fscalar,int,deg)

    # Perform action based on polynomial
    Action=zeros(n,m)
    for j=1:m
        x_result=zeros(n)
        potenz = Om[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:deg
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
    end
    Action=Action.*Om-(N*Om).*Om

    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    diag2 = sum(Action,dims=2)./normalization

    # Return result
    return diag1+diag2
end

function funDiagPP_Chebyshev(A, fAfun,fscalar,s_in,p,int,dist,normal=true,q=4)
    r = Int(round(s_in/3))
    m = s_in-2*r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
    U, S = nystrom(A, r, q)
    #t1 = sum(diagm(fscalar.(S)))
    N = U*diagm(fscalar.(S))*U'
    diag1=diag(N)

    # Stochastic trace estimation phase
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,m).-1
    elseif dist==:Gaussian
        Om=randn(n,m)
    elseif dist==:custom
        Om=testmatrix
    else
        print("Unknown distribution")
    end
    S = Chebyshev(int[1]..int[2])
    x = points(S, p)
    y = fscalar.(x)
    f_fit = Polynomials.fit(x, y);

    # Perform action based on polynomial
    Action=zeros(n,m)
    for j=1:m
        x_result=zeros(n)
        potenz = Om[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:p
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
    end
    Action=Action.*Om-(N*Om).*Om

    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    diag2 = sum(Action,dims=2)./normalization

    # Return result
    return diag1+diag2
end

"""
    Helper functions
"""

function KrylovMatfunc(A,b,f,m_Kr)
    Km=ExponentialUtilities.arnoldi(A,b./norm(b),m=m_Kr);
    result=vec(Km.V[:,1:m_Kr]*(f(Km.H[1:m_Kr,:])*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    return result
end

function PolyMatfunc(A,b,f,degree,int)
    # Approximate the function f polynomially
    S = Chebyshev(int[1]..int[2])
    x = points(S, degree)
    y = f.(x)
    f_fit = Polynomials.fit(x, y);
    potenz = b
    result = f_fit.coeffs[1]*potenz
    for i=2:degree
        potenz = A*potenz
        result = result + f_fit.coeffs[i]*potenz
    end
    return result
end

function RemezPoly(f,int,degree)
    rem=ratfn_minimax(f, int, degree, 0)
    return Polynomial(rem[1])
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
