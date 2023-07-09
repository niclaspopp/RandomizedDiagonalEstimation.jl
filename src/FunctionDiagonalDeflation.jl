function GirardHutchinsonFunctionDiagonalPolyApproxDeflation(A,s,k,f,p,int,dist=:Gaussian,normalization=true,testmatrix=nothing)
    # A: input matrix, s: number of samples, k: rank of low dimensional approx, p: degree of the polynomial approximation, dist: distribution for random vectors, normalization: select if normalization should be performed or not

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
    # Low rank approx
    # low rank approximation
    NysSketch=NystromSketch(Hermitian(A), k, k)
    FA_nys=NysSketch.U*diagm(f.(diag(NysSketch.Λ)))*(NysSketch.U)'
    diag_lr=diag(FA_nys)
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
    Action=Action.*Om-(FA_nys*Om).*Om

    #Calculate normalization
    if normalization==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    result = sum(Action,dims=2)./normalization

    return vec(diag_lr+result)
end

function GirardHutchinsonFunctionDiagonalKrylovDeflation(A,s,k,f,m_Kr,dist=:Gaussian,normalization=true,testmatrix=nothing)
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
    # # orthogonalize test matrix
    # if orthogonalize==true
    #     Q,R=qr(Omega)
    #     Om=Matrix(Q)
    # else
        Om=Omega
    # end
    # Low rank Sketch
    NysSketch=NystromSketch(Hermitian(A), k, k)
    FA_nys=NysSketch.U*diagm(f.(diag(NysSketch.Λ)))*(NysSketch.U)'
    diag_lr=diag(FA_nys)
    # # Perform action based on Krylov subspace
    Action=zeros(n,s)
    for j=1:s
        # Compute Krylov subspace
        Action[:,j]=KrylovMatfunc(A,Om[:,j],f,m_Kr)
    end
    #print(Action - Ainv*Om)
    Action=Action.*Om-(FA_nys*Om).*Om

    #Calculate normalization
    if normalization==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    result=zeros(n)
    if s>=1
        result = sum(Action,dims=2)./normalization
    end

    return vec(diag_lr+result)
end


function KrylovMatfunc(A,b,f,m_Kr)
    Km=ExponentialUtilities.arnoldi(A,b./norm(b),m=m_Kr);
    result=vec(Km.V[:,1:m_Kr]*(f(Km.H[1:m_Kr,:])*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    return result
end
