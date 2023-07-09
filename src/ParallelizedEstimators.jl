function GirardHutchinsonDiagonal_Parallel(A,s,dist=:Rademacher,normalization=true,O=nothing)
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

    Action = zeros(n,s)
    V = zeros(n,s)
    result = zeros(n)
    nthreads = Threads.nthreads()
    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:s
            Action[:,indices] = A*Om[:,indices].*Om[:,indices]
            V[:,indices] = Om[:,indices].*Om[:,indices]
            #print(Threads.threadid())
        end
    end

    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:n
            result[indices] = sum(Action[indices,:],dims=2)./sum(V[indices,:],dims=2)
            #print(Threads.threadid())
        end
    end

    return result
end


function DiagPP_Parallel(A,s_in,dist=:Rademacher,normalization=true,O=nothing)
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
    # mat-mat multipliciation should be parellelized automatically
    M=Q*(Q'*(A*Q))*Q'
    Action = zeros(n,s)
    V = zeros(n,s)
    result_stoch = zeros(n)

    nthreads = Threads.nthreads()
    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:s
            v=Om[:,indices]
            Action[:,indices] = A*(v-(Q*(Q'*v)))-Q*(Q'*(A*(v-(Q*(Q'*v)))))
            V[:,indices] = v.*v
            result_stoch[indices] = sum(Action[indices,:],dims=2)./sum(V[indices,:],dims=2)
            #print(Threads.threadid())
        end
    end


    return vec(diag(M)+result_stoch)
end


function GirardHutchinsonDiagonal_HutchinsonShift_Parallel(A,s,dist=:Rademacher,normalization=true,O=nothing)
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
    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:s
            for j=1:length(indices)
                A_approx = A_approx+1/s*Sketch[:,indices[j]]*(Omega[:,indices[j]])'
            end
        end
    end
    S = A_approx - diagm(diag(A_approx))
    Action = zeros(n,s)
    V = zeros(n,s)
    result = zeros(n)
    nthreads = Threads.nthreads()
    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:s
            Action = (Sketch[:,indices]-S*Omega[:,indices]).*Omega[:,indices]
            #Calculate normalization
            if normalization==true
                V = Omega[:,indices].*Omega[:,indices]
                normalization =  sum(V,dims=2)
            else
                normalization = s*ones(n)
            end
            result[indices] = sum(Action[indices,:],dims=2)./sum(V[indices,:],dims=2)
        end
    end
    # Calculate the final result
    result_final = sum(Action,dims=2)./normalization

    return vec(result_final)
end

function XDiag_Efficient_Parallel(A,s_in,dist=:Rademacher,O=nothing)
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
    diag_est=zeros(n,s)
    #QQt=Q*Q'
    QAQ = Q*(Q'*((A*Q)*Q'))
    nthreads = Threads.nthreads()
    Threads.@threads for i = 1:nthreads
        id = Threads.threadid()
        if id==i
            indices = id:nthreads:s
            for j=1:length(indices)
                omega_i =  Om[:,indices[j]]
                # low rank part
                s_i =  S[:,indices[j]]
                v_i = Q*s_i
                Av=A*v_i
                diag_est[:,indices[j]] += vec(diag(QAQ)-diag(v_i*(((v_i'*A)*Q)*Q'))-diag((Q*(Q'*(Av)))*v_i')+diag(v_i*(((v_i'*A)*v_i)*v_i')))
                # stochastic estimation part
                diag_est[:,indices[j]] += (omega_i.*(A*omega_i-(QAQ*omega_i-v_i*(v_i'*(A*(Q*(Q'*omega_i))))-Q*(Q'*(Av))*(v_i'*omega_i)+v_i*(v_i'*(Av))*(v_i'*omega_i))))./(omega_i.*omega_i)
            end
        end
    end


    return vec((s^-1)*sum(diag_est,dims=2))
end
