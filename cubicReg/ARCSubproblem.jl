function ARCSubproblem(gx,B,v1,lambda1,sigma,kappa_easy,subMaxIts) %#codegen
# Usage: p = ARCSubproblem(gx,B,v1,lambda1,sigma,kappa_easy,maxIts)
#adapted from Algorithm 7.3.6, Conn et al 2000, Algorithm 4.3, Nocedal
#and Wright, 2006, and Cartis et al 2011

#subout returns -1 if Hessian is ill-conditioned, 0 if maxIts is reached, 1
#if a good solution is found, 2 if an edge case is reached

lambda = max(0,-lambda1)
#small value to resolve numerical instability from eig and chol
if lambda != 0 #strictly greater than
    lambda = lambda + eps(lambda)
end
Delta = lambda/sigma

B1=deepcopy(B)

i=0

#we are trying to find the global minimizer of m(p). It should satisfy
#1. (B+lambda*I)p = -g
#2. lambda^2 = sigma^2*norm(p)
#see Eqn (6.1) in Cartis et al 2011
while true
    i=i+1
    #increment diagonal more efficiently than whole-matrix addition
    B1[1:1+size(B1,1):end] = diag(B).+lambda
    C = cholesky(B1,check=false)
    flag = LinearAlgebra.issuccess(C)
    k = 0
    lambda_const = eps(lambda)
    #fail to be positive definite
    while flag == 0
        k = k+1
        B1[1:1+size(B1,1):end] = diag(B1).+lambda_const
        C = cholesky(B1,check=false)
        flag = LinearAlgebra.issuccess(C)
        # gradually increase the constant to reduce iterations
        lambda_const = min(lambda_const*2,abs(lambda1))
        if k > subMaxIts
            p = zeros(size(gx))
            # -1 = failed to find cholesky
            return p,(info=-1,iter=i)
        end
    end
    p = -C.U\(C.L\gx)
    normp = norm(p)
    if i == subMaxIts
        # 0 = maxIts reached
        return p,(info=0,iter=i)
    end
    if abs(normp-Delta) < kappa_easy
        # 1 = good solution found
        return p,(info=1,iter=i)
    end
    if normp <= Delta
        if lambda == 0 || normp == Delta
            # 2 = edge case
            return p,(info=2,iter=i)
        else #"hard case", Nocedal and Wright, 2006 P88 Equation 4.45
            #%also see Algorithm 7.3.6, P199 of Conn 2000
            #fprintf('A wild "hard case" appeared!\n');
            #1*tau^2+0*tau+(norm^2-Delta^2)=0
            #normp is the same as the expression cuz the first eigenvector
            #dot gx is zero
            #hp = [1 0 norm(V(:,2:end)'*gx./(evals(2:end)+lambda))^2-Delta^2];
            # find the larger root of the quadratic equation
            tau = maximum(roots(Polynomial([1,0,normp^2-Delta^2])))
            p = tau*v1 + p
        end
    end
    q = C.L\p
    if lambda < eps(lambda)
        lambda = lambda + lambda_const
    else
        #Newton method to update lambda, simplified version of phi(lambda)/phi'(lambda) eqn (6.7) and (6.10) in Nocedal
        #and Wright
        lambda = lambda + lambda*(normp-Delta)/(normp+Delta*lambda*(norm(q)/normp)^2)
    end
    Delta = lambda/sigma
end
end
