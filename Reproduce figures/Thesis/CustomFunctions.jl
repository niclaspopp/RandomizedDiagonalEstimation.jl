
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

function KrylovMatfuncLog(A,b,f,m_Kr)
    Km=ExponentialUtilities.arnoldi(A,b./norm(b),m=m_Kr);
    print(Km.H[1:m_Kr,:])
    matwrite("ExampleHessenberg.mat", Dict(
	"H" => Km.H[1:m_Kr,:]))
    result=vec(Km.V[:,1:m_Kr]*(logmat(Matrix(Km.H[1:m_Kr,:])+Matrix(I,m_Kr,m_Kr))*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    return result
end

function logmat(A)
       Λ, S = eigen(A)
       return S*(log.(Complex.(Λ)).*inv(S))
end

function KrylovMatfuncCustom(A,b,f,m_Kr)
    V,H=arnoldi(A,b./norm(b),m_Kr,"double");
    result=vec(V[:,1:m_Kr]*(f(H[1:m_Kr,:])*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    return result
end

"""
Pure function estimators
"""

function KrylovMatfunc(A,b,f,m_Kr)
    Km=ExponentialUtilities.arnoldi(A,b./norm(b),m=m_Kr);
    f_k = f(Matrix(Km.H[1:m_Kr,:]))

    result=vec(Km.V[:,1:m_Kr]*(f_k*Matrix(I,m_Kr,m_Kr)[:,1])*norm(b))
    print(sum(isnan.(Km.V[:,1:m_Kr])),"\n")
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

function PolytoFunc(A,b,p,degree)
    # Approximate the function f polynomially
    potenz = b
    result = p.coeffs[1]*potenz
    for i=2:degree
        potenz = A*potenz
        result = result + p.coeffs[i]*potenz
    end
    return result
end

function ChebyshevFunc(f,degree,int)
    # Approximate the function f polynomially
    S = Chebyshev(int[1]..int[2])
    x = points(S, degree)
    y = f.(x)
    f_fit = Polynomials.fit(x, y);
    return f_fit
end


"""
    funNyström reproduction
"""

function funNyström(A,f,k,q=2)
    # get sizes of the matrix
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Sample test matrix
    Omega = randn(n,k)
    Q,R = qr(Omega)
    Q = Matrix(Q)
    # subspace iteration
    for i=1:q
        X = A*Q
        Q,R = qr(X)
        Q = Matrix(Q)
    end
    # decomposition
    Y=A*Q
    eigdec = eigen(Q'*Y)
    D=diagm(eigdec.values)
    V=Matrix(eigdec.vectors)
    B=Y*V*pinv(D)*V'
    U,S,Vt=svd(B)
    V=Matrix(Vt')
    L_vec=S.^2
    # final result
    return U*diagm(f.(L_vec))*U'
end

function funNyströmDiag(A,f,rk)
	NS2=NystromSketch(A, rk, rk)
	U22=NS2.U
	S22=NS2.Λ
	N2 = U22*diagm(f.(diag(S22)))*U22'
	return diag(N2)
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

"""
    funDiagPP
"""

function KrylovfunDiagPP(A, fAfun,fscalar,s_in,deg,q=4)

    r = Int(round(s_in/2))
    m = s_in-r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	# U, S = nystrom(A, r, q)
    # NS=NystromSketch(A_exp, r, r)
    # U=NS.U
    # S=diag(NS.Λ)
    # #t1 = sum(diagm(fscalar.(S)))
    # N = U*diagm(fscalar.(S))*U'
    # diag1=diag(N)
	NS2=NystromSketch(A, r, r)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
	diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega=2*rand(d,n,s).-1
    Action=zeros(n,m)
    for j=1:m
        # Compute Krylov subspace
        Action[:,j]=KrylovMatfunc(A,Omega[:,j],fAfun,deg)
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

function ChebyshevfunDiagPP(A, fAfun,fscalar,s_in,deg,int,q=4)

    r = Int(round(s_in/2))
    s = s_in-r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	# U, S = nystrom(A, r, q)
    # NS=NystromSketch(A_exp, r, r)
    # U=NS.U
    # S=diag(NS.Λ)
    # #t1 = sum(diagm(fscalar.(S)))
    # N = U*diagm(fscalar.(S))*U'
    # diag1=diag(N)
	NS2=NystromSketch(A, r, r)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
	diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega=2*rand(d,n,s).-1
    Action=zeros(n,s)

    # Poly approx to f(A)b
    S = Chebyshev(int[1]..int[2])
    x = points(S, deg)
    y = fscalar.(x)
    f_fit = Polynomials.fit(x, y);

    # Perform action based on polynomial
    Action=zeros(n,s)
    for j=1:s
        x_result=zeros(n)
        potenz = Omega[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:deg
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
    end
    # Y = U' * Omega
    diag2=zeros(n)
    for j=1:s
        # Compute Krylov subspace
        diag2+=(Action[:,j].*Omega[:,j]-(N*Omega[:,j]).*Omega[:,j]) / s
    end
    #t2 = (tr(Omega' * Action) - tr(Y' * diagm(fscalar.(S)) * Y)) / m

    # Return result
    return diag1+diag2
end

function RemezfunDiagPP(A, fAfun,fscalar,s_in,deg,int,q=4)

    r = Int(round(s_in/2))
    s = s_in-r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	# U, S = nystrom(A, r, q)
    # NS=NystromSketch(A_exp, r, r)
    # U=NS.U
    # S=diag(NS.Λ)
    # #t1 = sum(diagm(fscalar.(S)))
    # N = U*diagm(fscalar.(S))*U'
    # diag1=diag(N)
	NS2=NystromSketch(A, r, r)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
	diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega=2*rand(d,n,s).-1
    Action=zeros(n,s)

    # Poly approx to f(A)b
    f_fit = RemezPoly(fscalar,int,deg)

    # Perform action based on polynomial
    for j=1:s
        x_result=zeros(n)
        potenz = Omega[:,j]
        x_result = f_fit.coeffs[1]*potenz
        for i=2:deg
            potenz = A*potenz
            x_result = x_result + f_fit.coeffs[i]*potenz
        end
        Action[:,j]=x_result
    end
    # determine stochastic estimate
    diag2=zeros(n)
    for j=1:s
        # Compute Krylov subspace
        diag2+=(Action[:,j].*Omega[:,j]-(N*Omega[:,j]).*Omega[:,j]) / s
    end

    # Return result
    return diag1+diag2
end

# using own implementation of the Arnoldi method
function KrylovfunDiagPPcustom(A, fAfun,fscalar,s_in,deg,q=4)

    r = Int(round(s_in/3))
    m = s_in-2*r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
	# U, S = nystrom(A, r, q)
    # NS=NystromSketch(A_exp, r, r)
    # U=NS.U
    # S=diag(NS.Λ)
    # #t1 = sum(diagm(fscalar.(S)))
    # N = U*diagm(fscalar.(S))*U'
    # diag1=diag(N)
	NS2=NystromSketch(A, r, r)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
	diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega=2*rand(d,n,m).-1
    #Omega=randn(n,m)
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

# function GirardHutchinsonFunctionDiagonalKrylovCustom(A,s_in,fAfun,deg)
#
#     m = s_in
#     # Implementation of Algorithm 2
#     (mA,n)=size(A)
#     if n!=mA
#         print("Matrix must be square")
#     end
#
#     # Stochastic trace estimation phase
#     d = Bernoulli(.5)
#     Omega=2*rand(d,n,m).-1
#     #Omega=randn(n,m)
#     Action=zeros(n,m)
#     for j=1:m
#         # Compute Krylov subspace
#         Action[:,j]=KrylovMatfuncCustom(A,Omega[:,j],fAfun,deg)
#     end
#     # Y = U' * Omega
#     diag2=zeros(n)
#     for j=1:m
#         # Compute Krylov subspace
#         diag2+=(Action[:,j].*Omega[:,j]) / m
#     end
#
#     # Return result
#     return diag2
# end

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
"""
    Exponential
"""

function funDiagPPExp(A,fscalar,s_in,deg,q=4)

    r = Int(round(s_in/3))
    m = s_in-2*r
    # Implementation of Algorithm 2
    (mA,n)=size(A)
    if n!=mA
        print("Matrix must be square")
    end
    # Low rank approximation phase
    # U, S = nystrom(A, r, q)
    # NS=NystromSketch(A_exp, r, r)
    # U=NS.U
    # S=diag(NS.Λ)
    # t1 = sum(diagm(fscalar.(S)))
	NS2=NystromSketch(A, r, r)
	U22=NS2.U
	S22=NS2.Λ
	N = U22*diagm(fscalar.(diag(S22)))*U22'
    diag1=diag(N)

    # Stochastic trace estimation phase
    d = Bernoulli(.5)
    Omega = 2*rand(d,n,m).-1
    Action=zeros(n,m)
    for j=1:m
        # Compute Krylov subspace
        Action[:,j]=expv(1,A,Omega[:,j],m=deg)-Omega[:,j]
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

function GirardHutchinsonFunctionDiagonalExpNorm(A,s,m_Kr,dist=:Gaussian,normal=true,orthogonalize=false,testmatrix=nothing)
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
        Action[:,j]=expv(1,A,Omega[:,j],m=m_Kr)-Omega[:,j]
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

"""
    Own arnoldi method
"""
function arnoldi(A,b,m,o)

    n=length(b);
    Q=zeros(n,m+1);
    H=zeros(m+1,m);
    Q[:,1]=b/norm(b)

    for k=1:m
        w=A*Q[:,k]; # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        # Implement this function or replace call with code for orthogonalization
        if o == "classical"
            h,β,z=my_hw1_gs_classical(Q,w,k);
        elseif o == "modified"
            h,β,z=my_hw1_gs_modified(Q,w,k);
        elseif o == "double"
            h,β,z=my_hw1_gs_double(Q,w,k);
        elseif o == "triple"
            h,β,z=my_hw1_gs_triple(Q,w,k);
        else
            print("Unknown GS Option")
        end
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    H[H.<5*10^(-16)].=0
    return Q,H
end

# Classical Gram-Schmidt
function my_hw1_gs_classical(Q,w,k)
    # Matrix Vector Product
    h=Q[:,1:k]'*w
    y=w-Q[:,1:k]*h
    return h,norm(y),y
end

# Modified Gram-Schmidt
function my_hw1_gs_modified(Q,w,k)
    # For Loop
    y=w
    h=zeros(k)
    for i in 1:k
        h[i]=Q[:,i]'y
        y=y-Q[:,i]*h[i]
    end
    return h,norm(y),y
end

# Double Gram-Schmidt
function my_hw1_gs_double(Q,w,k)
    h,β,y=my_hw1_gs_classical(Q,w,k);
    g,β,y=my_hw1_gs_classical(Q,y,k);
    return h+g,norm(y),y
end

# Triple Gram-Schmidt
function my_hw1_gs_triple(Q,w,k)
    h,β,y=my_hw1_gs_classical(Q,w,k);
    g,β,y=my_hw1_gs_classical(Q,y,k);
    e,β,y=my_hw1_gs_classical(Q,y,k);
    return h+g+e,norm(y),y
end
