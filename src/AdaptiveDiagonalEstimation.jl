function ADiagPP(A, epsilon, delta,con=1,max_iter=0)

    # Set up parameters
    matrix_size = size(A)
    n = Int(matrix_size[1])

    if max_iter == 0
        max_iter = Int(n)
    end

    Afun(x) = A*x
    C = 4*con*log(2/delta)/epsilon^2  # Constant for number of matvecs

    flag = true
    fnc = []

    # First iteration
    y = Afun(randn(n))
    q = y / norm(y)
    Q = q
    x = Afun(q)
    t = q' * x
    ############
    ## TO DO: ADJUST D
    ## d = tr(q*(q'*(A*q)*q'))
    Actions = A*q
    d = diag(q*(q'*Actions)*q')
    c = t^2
    #trest1 = t
    diagest1 = d
    iteration = 1
    b = norm(x)^2
    append!(fnc,2 * iteration + C .* (c .- 2 * b))

    # Remaining iterations
    while flag
        # Get new column in Q
        y = Afun(randn(n))
        qt = y - Q * (Q' * y)
        if norm(qt) < 1e-10

            lowrank_matvecs = 2 * iteration
            trest_matvecs = 0
            total_matvecs = lowrank_matvecs + trest_matvecs
            #trest = trest1
            diagest1 = diagest1

        end
        qt = qt / norm(qt)
        qt = qt - Q * (Q' * qt)
        q = qt / norm(qt)
        Q = [Q q]
        x = Afun(q)
        Actions=hcat(Actions,x)
        # Update recursion
        b = b + norm(x)^2
        t = q' * x
        diagest1 = diag(Q*(Q'*Actions)*Q')
        c = c + 2 * norm(Q[:, 1:iteration]' * x)^2 + t^2

        # Update function
        iteration = iteration + 1
        append!(fnc,2 * iteration + C * (c - 2 * b))
        # Check if iteration should be stopped
        if (iteration > 2) && (fnc[iteration - 1] < fnc[iteration]) && (fnc[iteration - 2] < fnc[iteration - 1])
            flag = false
        end
        #print(diagest1,"\n")
    end

    lowrank_matvecs = 2 * iteration

    # Combine Hutchinson and Frobenius norm estimation
    iteration = 0
    flag = true
    t=0
    num_arr = zeros(n)
    denom_arr = zeros(n)

    while flag

        psi = randn(n)
        y = psi - Q * (Q' * psi)
        y = Afun(y)
        y = y - Q * (Q' * y)
        num_arr += y.*psi
        denom_arr += psi.*psi
        t = t + y' * y
        estFrob = t / (iteration + 1)
        alpha = supfind(iteration + 1, delta)
        M = ceil(C * estFrob / alpha)
        if iteration + 1 > M
            flag = false
        end
        iteration = iteration + 1
    end
    # Perform Hutchinson estimation
    diagest2 = num_arr./denom_arr
    # Print number of iterations
    diagest_matvecs = iteration
    print("Total matvecs: ", lowrank_matvecs + diagest_matvecs,"\n")

    # Return outputs
    return diagest=diagest1+diagest2;
end


function supfind(i,d)
    f(a) = gamma_inc(i/2,a*i/2)[1]
    alphalist = reverse(0:0.01:1)
    fa = 1
    index = 1
    while fa > d && index<length(alphalist)
        fa = f(alphalist[index])
        index += 1
    end
    alpha = alphalist[index]
    return alpha
end
