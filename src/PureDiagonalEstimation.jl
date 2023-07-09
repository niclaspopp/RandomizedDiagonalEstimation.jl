function GirardHutchinsonDiagonal(A,s,dist=:Rademacher,normalization=true,O=nothing)
    # A: input matrix, s: number of samples, dist: distribution for random vectors, normalization: select if normalization should be performed or not

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
        Om=O
    else
        print("Unknown distribution")
    end
    # Perform action of A
    Action = (A*Om).*Om
    #Calculate normalization
    if normalization==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    result = sum(Action,dims=2)./normalization

    return vec(result)
end

function DiagPP(A,s_in,dist=:Rademacher,normalization=true,O=nothing)
    #A: input matrix, s: nof samples overall
    k = Int(round(s_in/3))
    s = s_in-2*k
    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,2*s).-1
    elseif dist==:Gaussian
        Om=randn(n,2*s)
    elseif dist==:custom
        Om=O
    else
        print("Unknown distribution")
    end
    M=zeros(size(A))
    T = A*Om[:,1:s]
    Q,R_d = qr(T)
    Q=Matrix(Q)
    d_vec = zeros(n)
    d_vec2 = zeros(n)
    for i=1:s
        v=Om[:,s+i]
        action = A*(v-(Q*(Q'*v)))-Q*(Q'*(A*(v-(Q*(Q'*v)))))
        d_vec += v.*action
        d_vec2+=v.*v
    end
    M=Q*(Q'*(A*Q))*Q'
    return vec(diag(M)+d_vec.*(d_vec2.^(-1)))
end

function GirardHutchinsonDiagonal_HutchinsonShift(A,s,dist=:Rademacher,normalization=true,O=nothing)
    # A: input matrix, s: number of samples, dist: distribution for random vectors, normalization: select if normalization should be performed or not

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    if dist==:Rademacher
        d = Bernoulli(.5)
        Omega=2*rand(d,n,s).-1
    elseif dist==:Gaussian
        Omega=randn(n,s)
    elseif dist==:custom
        Omega=O
    else
        print("Unknown distribution")
    end
    Q,R=qr(Omega)
    Omega=Matrix(Q)
    # Sketching
    Sketch = (A*Omega)
    # Full approximation of A
    A_approx = zeros(n,n)
    for i=1:s
        A_approx = A_approx+1/s*Sketch[:,i]*(Omega[:,i])'
    end
    # Shift matrix
    S = A_approx - diagm(diag(A_approx))
    # Perform action of A
    Action = (Sketch-S*Omega).*Omega
    #Calculate normalization
    if normalization==true
        V = Omega.*Omega
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    result = sum(Action,dims=2)./normalization

    return result
end

function NysDiagPP(A,s,dist=:Rademacher,normalization=true,O=nothing)
    #A: input matrix, s: nof samples for diagonal estimation

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    k = Int(round(s/2))
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,k).-1
    elseif dist==:Gaussian
        Om=randn(n,k)
    elseif dist==:custom
        Om=O
    else
        print("Unknown distribution")
    end
    # low rank approximation
    NysSketch=NystromSketch(Hermitian(A), k, k)
    A_nys=NysSketch.U*Matrix(NysSketch.Λ)*(NysSketch.U)'
    diag_lr=diag(A_nys)
    # Perform action of A
    Action = (A*Om-A_nys*Om).*Om
    #Calculate normalization
    if normalization==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the stochastic result
    diag_stoch = sum(Action,dims=2)./normalization

    return vec(diag_lr+diag_stoch)
end

function RSVD_Diag(A,s_in,dist=:Rademacher,O=nothing)
    #A: input matrix, s: nof samples for diagonal estimation
    s = Int(round(s_in/2))
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
        Om=O
    else
        print("Unknown distribution")
    end
    # low rank approximation
    T = A*Om
    Q,R_d = qr(T)
    Q=Matrix(Q)
    diag_lr = diag(Q*(Q'*(A*Q))*Q')
    return diag_lr
end

function XDiag(A,s_in,dist=:Rademacher,O=nothing)
    #A: input matrix, s: nofmatvec
    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    k = Int(round(s_in/2))
    s = s_in-k
    # Sample random test matrix
    if dist==:Rademacher
        d = Bernoulli(.5)
        Omega=2*rand(d,n,s).-1
    elseif dist==:Gaussian
        Omega=randn(n,s)
    elseif dist==:custom
        Omega=O
    else
        print("Unknown distribution")
    end
    # A: input matrix, s: number of samples
    T = A*Omega
    diag_est = zeros(n)
    for i = 1:s
        # select i-th column
        omega_i =  Omega[:,i]
        # low rank part
        T_i = T[:,1:end .!=i]
        Q_it,R_i = qr(T_i)
        Q_i=Matrix(Q_it)
        diag_est += vec(diag(Q_i*(Q_i'*A)))
        # stochastic estimation part
        diag_est += (omega_i.*(A*omega_i-Q_i*(Q_i'*(A*omega_i))))./(omega_i.*omega_i)
    end
    return (s^-1)*diag_est
end

function XDiag_Efficient(A,s_in,dist=:Rademacher,O=nothing)
    # A: input matrix, s: number of matvecs, dist: distribution for random vectors, normalization: select if normalization should be performed or not
    k = Int(round(s_in/2))
    s = s_in-k
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
        Om=O
    else
        print("Unknown distribution")
    end
    # Sketching
    Y = A*Om
    Q,R=qr(Y)
    Q=Matrix(Q)
    # Calculate rank 1 update vectors
    S=(R')\Matrix(I,s,s)
    foreach(normalize!, eachcol(S))
    # perform diagonal estimation
    diag_est=zeros(n)
    #QQt=Q*Q'
    QAQ = Q*(Q'*((A*Q)*Q'))
    for i = 1:s
        # select i-th column
        omega_i =  Om[:,i]
        # low rank part
        s_i =  S[:,i]
        v_i = Q*s_i
        #QiQi_s = QQt-(((Q*s_i)*s_i')*Q')
        Av=A*v_i
        diag_est += vec(diag(QAQ)-diag(v_i*(((v_i'*A)*Q)*Q'))-diag((Q*(Q'*(Av)))*v_i')+diag(v_i*(((v_i'*A)*v_i)*v_i')))
        # stochastic estimation part
        diag_est += (omega_i.*(A*omega_i-(QAQ*omega_i-v_i*(v_i'*(A*(Q*(Q'*omega_i))))-Q*(Q'*(Av))*(v_i'*omega_i)+v_i*(v_i'*(Av))*(v_i'*omega_i))))./(omega_i.*omega_i)
    end
    return (s^-1)*diag_est
end

function XDiag_old(A, s_in)
    # A: input matrix, s: number of samples
    s = Int(round(s_in/2))
    (m,n)=size(A)
    d = Bernoulli(.5)
    Omega=2*rand(d,n,s).-1
    T = A*Omega
    diag_est = zeros(n)
    for i = 1:s
        # select i-th column
        omega_i =  Omega[:,i]
        # low rank part
        T_i = T[:,1:end .!=i]
        Q_it,R_i = qr(T_i)
        Q_i=Matrix(Q_it)
        diag_est += vec(diag(Q_i*(Q_i'*A)))
        # stochastic estimation part
        diag_est += (omega_i.*(A*omega_i-Q_i*(Q_i'*(A*omega_i))))./(omega_i.*omega_i)
    end
    return (s^-1)*diag_est
end
