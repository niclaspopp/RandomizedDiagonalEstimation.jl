function MoM_Permutation(M,k,l)
    (m,n)=size(M)
    Means = zeros(m,k)
    for h=1:k
        subset = randperm(n)[1:l]
        Means[:,h]=mean(M[:,subset],dims=2)
    end
    final_result=median(Means,dims=2)
    return vec(final_result)
end


function GirardHutchinsonDiagonalMoM(A,s,ngroups,groupsize,dist=:Rademacher,normalization=true,O=nothing)
    # A: input matrix, s: number of samples, dist: distribution for random vectors, normalization: select if normalization should be performed or not

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    Om=zeros(n,s)
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
    else
        V = ones(n,d)
    end
    # Calculate the final result
    result = MoM_Permutation(Action,ngroups,groupsize)./MoM_Permutation(V,ngroups,groupsize)

    return vec(result)
end

function DiagPPMoM(A,s_in,ngroups,groupsize,dist=:Rademacher,normalization=true,O=nothing)
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
    d_vec = zeros(n,s)
    d_vec2 = zeros(n,s)
    for i=1:s
        v=Om[:,s+i]
        action = A*(v-(Q*(Q'*v)))-Q*(Q'*(A*(v-(Q*(Q'*v)))))
        d_vec[:,i]= v.*action
        d_vec2[:,i]=v.*v
    end
    M=Q*(Q'*(A*Q))*Q'
    return vec(diag(M)+MoM_Permutation(d_vec,ngroups,groupsize)./MoM_Permutation(d_vec2,ngroups,groupsize))
end

function GirardHutchinsonDiagonal_HutchinsonShiftMoM(A,s,ngroups,groupsize,dist=:Rademacher,normalization=true,O=nothing)
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
    else
        V = ones(n,d)
    end
    # Calculate the final result
    result = MoM_Permutation(Action,ngroups,groupsize)./MoM_Permutation(V,ngroups,groupsize)

    return vec(result)
end

function NysDiagPPMoM(A,s,ngroups,groupsize,dist=:Rademacher,normalization=true,O=nothing)
    #A: input matrix, s: nof samples for diagonal estimation

    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    Om=zeros(n,s)
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
    NysSketch=NystromSketch(A, k, k)
    A_nys=NysSketch.U*Matrix(NysSketch.Λ)*(NysSketch.U)'
    diag_lr=diag(A_nys)
    # Perform action of A
    Action = (A*Om-A_nys*Om).*Om
    #Calculate normalization
    if normalization==true
        V = Om.*Om
    else
        V = ones(n,d)
    end
    # Calculate the final result
    diag_stoch = MoM_Permutation(Action,ngroups,groupsize)./MoM_Permutation(V,ngroups,groupsize)

    return vec(diag_lr+diag_stoch)
end


function XDiagMoM(A,s_in,ngroups,groupsize,dist=:Rademacher,O=nothing)
    #A: input matrix, s: nofmatvec
    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    k = Int(round(s_in/2))
    s = s_in-k
    # Sample random test matrix
    Om=zeros(n,s)
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
    diag_est = zeros(n,s)
    for i = 1:s
        # select i-th column
        omega_i =  Omega[:,i]
        # low rank part
        T_i = T[:,1:end .!=i]
        Q_it,R_i = qr(T_i)
        Q_i=Matrix(Q_it)
        diag_est[:,i]+= vec(diag(Q_i*(Q_i'*A)))
        # stochastic estimation part
        diag_est[:,i]+= (omega_i.*(A*omega_i-Q_i*(Q_i'*(A*omega_i))))./(omega_i.*omega_i)
    end
    return MoM_Permutation(diag_est,ngroups,groupsize)
end

function XDiag_EfficientMoM(A,s_in,ngroups,groupsize,dist=:Rademacher,O=nothing)
    # A: input matrix, s: number of matvecs, dist: distribution for random vectors, normalization: select if normalization should be performed or not
    k = Int(round(s_in/2))
    s = s_in-k
    # get sizes of the matrix
    (m,n)=size(A)
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    Om=zeros(n,s)
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
    diag_est=zeros(n,s)
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
        diag_est[:,i] += vec(diag(QAQ)-diag(v_i*(((v_i'*A)*Q)*Q'))-diag((Q*(Q'*(Av)))*v_i')+diag(v_i*(((v_i'*A)*v_i)*v_i')))
        # stochastic estimation part
        diag_est[:,i] += (omega_i.*(A*omega_i-(QAQ*omega_i-v_i*(v_i'*(A*(Q*(Q'*omega_i))))-Q*(Q'*(Av))*(v_i'*omega_i)+v_i*(v_i'*(Av))*(v_i'*omega_i))))./(omega_i.*omega_i)
    end
    return MoM_Permutation(diag_est,ngroups,groupsize)
end
