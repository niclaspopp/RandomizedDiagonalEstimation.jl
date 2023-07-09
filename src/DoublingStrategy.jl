function GirardHutchinsonDiagonal_Doubling(A,s_start,var,dist=:Rademacher,normal=true)
    # A: input matrix, s: number of samples, dist: distribution for random vectors, normal: select if normalization should be performed or not

    # get sizes of the matrix
    (m,n)=size(A)
    # sample start matrix
    s=s_start
    if n!=m
        print("Matrix must be square")
    end
    # Sample random test matrix
    if dist==:Rademacher
        d = Bernoulli(.5)
        Om=2*rand(d,n,s).-1
    elseif dist==:Gaussian
        Om=randn(n,s)
    else
        print("Unknown distribution (custom distributions are not available when using the doubling strategy)")
    end
    # Initialize array
    Action = zeros(n,s)
    V=zeros(n,s)
    normailzation=zeros(n)
    D_s = zeros(n)
    err_est = 0
    ### Fist step

    # Perform action of A
    Action = (A*Om).*Om
    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        V=ones(n,s)
        normalization = s*ones(n)
    end
    # Calculate the diagonal estimator result
    D_s = sum(Action,dims=2)./normalization
    # Calculate the variance
    Single_estimates = Action./V
    Single_estimate_error = Single_estimates .- D_s
    Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
    err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    #print(err_est)
    doubling_steps=0
    while err_est>var
        doubling_steps+=1
        # Sample another random test matrix
        if dist==:Rademacher
            d = Bernoulli(.5)
            Om_add=2*rand(d,n,s).-1
        elseif dist==:Gaussian
            Om_add=randn(n,s)
        elseif dist==:custom
            Om_add=O
        end
        #Om=cat(Om,Om_add,dims=2)
        # Perform action of A
        Action = cat(Action,(A*Om_add).*Om_add,dims=2)
        #Calculate normalization
        if normal==true
            V = cat(V,Om_add.*Om_add,dims=2)
            normalization =  sum(V,dims=2)
        else
            V=ones(n,2*s)
            normalization = s*ones(n)
        end
        # Calculate the diagonal estimator result --> recalculated completely
        D_s = sum(Action,dims=2)./normalization
        # Calculate the variance --> only add updates
        Single_estimates = cat(Single_estimates,(Action[:,s+1:end]./V[:,s+1:end]),dims=2)
        Single_estimate_error = Single_estimates .- D_s
        Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
        s=2*s
        err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    end
    return D_s,err_est,doubling_steps
end


function DiagPP_Doubling(A,s_start,var,dist=:Rademacher,normal=true)
    # A: input matrix, s: number of samples, dist: distribution for random vectors, normal: select if normalization should be performed or not
    k = Int(round(s_start/3))
    s = s_start-2*k
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
    M=Q*(Q'*(A*Q))*Q'



    # Initialize array
    Action = zeros(n,s)
    d_vec = zeros(n,s)
    d_vec2 = zeros(n,s)
    d_est = zeros(n,s)
    V=zeros(n,s)
    normalization=zeros(n)
    D_s = zeros(n)
    err_est = 0
    ### Fist step

    # Perform action of A
    for i=1:s
        v=Om[:,s+i]
        Action[:,i] = A*(v-(Q*(Q'*v)))-Q*(Q'*(A*(v-(Q*(Q'*v)))))
        d_vec[:,i]= v.*Action[:,i]
        d_vec2[:,i]=v.*v
        d_est[:,i]=vec(diag(M)+d_vec[:,i].*(d_vec2[:,i].^(-1)))
    end
    #Calculate normalization
    if normal==true
        V = d_vec2
        normalization =  sum(V,dims=2)
    else
        V=ones(n,s)
        normalization = s*ones(n)
    end
    # Calculate the diagonal estimator result
    D_s = vec(diag(M)+sum(d_vec,dims=2)./normalization)
    # Calculate the variance
    Single_estimate_error = d_est .- D_s
    Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
    err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    doubling_steps=0

    while err_est>var
        doubling_steps+=1
        # Sample another random test matrix
        if dist==:Rademacher
            d = Bernoulli(.5)
            Om_add=2*rand(d,n,s).-1
        elseif dist==:Gaussian
            Om_add=randn(n,s)
        elseif dist==:custom
            Om_add=O
        end
        # Remove influence of "old" M
        d_est = d_est .- vec(diag(M))
        Om = cat(Om,Om_add,dims=2)
        Om_lr = cat(Om[:,1:s],Om_add[:,1:s],dims=2)
        T = A*Om_lr
        Q,R_d = qr(T)
        Q=Matrix(Q)
        M=Q*(Q'*(A*Q))*Q'
        # Add inluence of "new" M
        d_est = d_est .+ vec(diag(M))
        # Initialize array
        Action = cat(Action,zeros(n,s),dims=2)
        V=cat(Action,zeros(n,s),dims=2)
        d_vec=cat(d_vec,zeros(n,s),dims=2)
        d_vec2=cat(d_vec2,zeros(n,s),dims=2)
        d_ext=cat(d_ext,zeros(n,s),dims=2)

        # Perform newly added
        for i=1:s
            v=Om_add[:,s+i]
            Action[:,s+i] = A*(v-(Q*(Q'*v)))-Q*(Q'*(A*(v-(Q*(Q'*v)))))
            d_vec[:,s+i]= v.*Action[:,i]
            d_vec2[:,s+i]=v.*v
            d_est[:,s+i]=vec(diag(M)+d_vec[:,i].*(d_vec2[:,i].^(-1)))
        end

        #Calculate normalization
        if normal==true
            V = d_vec2
            normalization =  sum(V,dims=2)
        else
            V=ones(n,2*s)
            normalization = s*ones(n)
        end
        # Calculate the diagonal estimator result --> recalculated completely
        D_s = vec(diag(M)+sum(d_vec,dims=2)./normalization)
        # Calculate the variance --> only add updates
        Single_estimates = cat(Single_estimates,(Action[:,s+1:end]./V[:,s+1:end]),dims=2)
        Single_estimate_error = Single_estimates .- D_s
        Var_sq = sum(sum(Single_estimate_error.^2,dims=2))

        s=2*s
        k=2*s
        err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    end
    return D_s,err_est,doubling_steps

end

function GirardHutchinsonDiagonalInverseCGDoubling_store(A,s,c_it,var,dist=:Gaussian,normal=true,testmatrix=nothing)
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
    D_s = sum(Action,dims=2)./normalization
    # Calculate variance
    Single_estimates = Action./V
    Single_estimate_error = Single_estimates .- D_s
    Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
    err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    err_est = err_est/norm(D_s)^2
    err_est_store = [err_est]
    #print(err_est)
    doubling_steps=0
    while err_est>var
        doubling_steps+=1
        # Sample another random test matrix
        if dist==:Rademacher
            d = Bernoulli(.5)
            Om_add=2*rand(d,n,s).-1
        elseif dist==:Gaussian
            Om_add=randn(n,s)
        elseif dist==:custom
            Om_add=O
        end

        # Perform action of A^-1 using CG
        Action = cat(Action,zeros(n,s),dims=2)
        for i=1:s
            Action[:,s+i] = cg(A,float.(Om_add[:,i]),itmax=c_it)[1].*Om_add[:,i]
        end
        #Calculate normalization
        if normal==true
            V = cat(V,Om_add.*Om_add,dims=2)
            normalization =  sum(V,dims=2)
        else
            V=ones(n,2*s)
            normalization = s*ones(n)
        end
        # Calculate the diagonal estimator result --> recalculated completely
        D_s = sum(Action,dims=2)./normalization
        # Calculate the variance --> only add updates
        Single_estimates = cat(Single_estimates,(Action[:,s+1:end]./V[:,s+1:end]),dims=2)
        Single_estimate_error = Single_estimates .- D_s
        Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
        s=2*s
        err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
        err_est_store = [err_est_store,err_est]
        print(err_est,"\n")
    end

    return vec(D_s),err_est_store
end

function GHDIPreCGDoubling_store(A,s,c_it,Sinv,var,dist=:Gaussian,normal=true,testmatrix=nothing)
    # A: input matrix, s: number of samples, Sinv: Preconditioner, c_it: number of iteration for conjugate gradient, dist: distribution for random vectors, normalization: select if normalization should be performed or not

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
        #Action[:,i] = cg(A,float.(Om[:,i]),itmax=c_it)[1].*Om[:,i]
        Action[:,i]=cg_preconditioned_complex(A,Sinv,float.(Om[:,i]),c_it).*Om[:,i]
    end

    #Calculate normalization
    if normal==true
        V = Om.*Om
        normalization =  sum(V,dims=2)
    else
        normalization = s*ones(n)
    end
    # Calculate the final result
    D_s = sum(Action,dims=2)./normalization
    # Calculate variance
    Single_estimates = Action./V
    Single_estimate_error = Single_estimates .- D_s
    Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
    err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)/norm(D_s)
    err_est = err_est
    err_est_store = [err_est]
    #print(err_est)
    doubling_steps=0
    while err_est>var
        doubling_steps+=1
        # Sample another random test matrix
        if dist==:Rademacher
            d = Bernoulli(.5)
            Om_add=2*rand(d,n,s).-1
        elseif dist==:Gaussian
            Om_add=randn(n,s)
        elseif dist==:custom
            Om_add=O
        end

        # Perform action of A^-1 using CG
        Action = cat(Action,zeros(n,s),dims=2)
        for i=1:s
            #Action[:,s+i] = cg(A,float.(Om_add[:,i]),itmax=c_it)[1].*Om_add[:,i]
            Action[:,s+i]=cg_preconditioned_complex(A,Sinv,float.(Om_add[:,i]),c_it).*Om_add[:,i]
        end
        #Calculate normalization
        if normal==true
            V = cat(V,Om_add.*Om_add,dims=2)
            normalization =  sum(V,dims=2)
        else
            V=ones(n,2*s)
            normalization = s*ones(n)
        end
        # Calculate the diagonal estimator result --> recalculated completely
        D_s = sum(Action,dims=2)./normalization
        # Calculate the variance --> only add updates
        Single_estimates = cat(Single_estimates,(Action[:,s+1:end]./V[:,s+1:end]),dims=2)
        Single_estimate_error = Single_estimates .- D_s
        Var_sq = sum(sum(Single_estimate_error.^2,dims=2))
        s=2*s
        err_est = sqrt(1/(s*(s-1)))*sqrt(Var_sq)
        err_est_store = [err_est_store,err_est]
        print(err_est,"\n")
    end

    return vec(D_s),err_est_store
end

function GHDIPreCG_storevar(A,s,c_it,Sinv,dist=:Gaussian,normal=true,testmatrix=nothing)
    # A: input matrix, s: number of samples, Sinv: Preconditioner, c_it: number of iteration for conjugate gradient, dist: distribution for random vectors, normalization: select if normalization should be performed or not

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
        #Action[:,i] = cg(A,float.(Om[:,i]),itmax=c_it)[1].*Om[:,i]
        Om[:,i] = Om[:,i]/norm(Om[:,i])
        Action[:,i]=cg_preconditioned_complex(A,Sinv,float.(Om[:,i]),c_it).*Om[:,i]
    end

    #Calculate normalization
    # if normal==true
    #     V = Om.*Om
    #     normalization =  sum(V,dims=2)
    # else
    normalization = s*ones(n)
    # end
    # Calculate the final result
    D_s = 1/s*sum(Action,dims=2)
    # Calculate variance
    Single_estimates = Action
    Single_estimate_error = Single_estimates .- D_s
    err_norms_sq = zeros(s)
    for i=1:s
        err_norms_sq[i] = norm(Single_estimate_error[:,i])^2
    end
    err_est = sqrt(1/(s*(s-1))*sum(err_norms_sq))
    return vec(D_s),err_est
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
