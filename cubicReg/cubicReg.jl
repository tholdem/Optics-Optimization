#output structs
mutable struct Output
    errHistory::Array{Float64,1}
    gradnorm::Array{Float64,1}
    xnorm::Array{Float64,1}
    sigma::Array{Float64,1}
    rho::Array{Float64,1}
    info::Int
    iter::Int
    subout::Array{NamedTuple,1}
end
function symmetricADHessian(f,x)
    Hx = ForwardDiff.hessian(f,x)
    return LinearAlgebra.tril(Hx,-1)+LinearAlgebra.tril(Hx)'
end
function cubicReg(f,x;grad=x->ForwardDiff.gradient(f,x),H=x->symmetricADHessian(f,x),errFcn=x->f(x), errTol=1e-10,maxIts=1e4,subMaxIts=500,sigma=1,sigma_min=1e-3,eta1=0.1,eta2=0.9,kappa_easy=1e-4)

    #m-function to compute rho
    m(p,gx,Hx,sigma) = dot(p,gx) + 0.5 * dot(Hx*p,p) + 1/3*sigma*norm(p)^3
    fx = f(x)
    gx = grad(x)

    out = Output([],[],[],[],[],-9,-9,[])
    push!(out.errHistory,errFcn(x))
    push!(out.gradnorm,norm(gx))
    push!(out.xnorm,norm(x))

    Hx = H(x)
    E = eigen(Hx)
    #leftmost eigenvalue/vector
    v1 = E.vectors[:,1]
    lambda1 = E.values[1]
    i=0
    while true
        i=i+1
        push!(out.sigma,sigma)
        p,subout = ARCSubproblem(gx,Hx,v1,lambda1,sigma,kappa_easy,subMaxIts)
        push!(out.subout,subout)
        #failed cholesky, terminate
        if subout.info == -1
            out.info = -1;
            out.iter = i-1;
            return x,out
        end
        rho = (fx-f(x+p))/(-m(p,gx,Hx,sigma))
        push!(out.rho,rho)
        #reach maxIts
        if i == maxIts
            out.info  = 0
            out.iter = maxIts
            return x,out
        end
        if rho >= eta1
            x = x + p
            #@printf "norm(x)=%f\n",norm(x)
            fx = f(x)
            gx = grad(x)
            #stopping-criteria reached
            if abs(fx) < errTol
                push!(out.errHistory,errFcn(x))
                push!(out.gradnorm,norm(gx))
                push!(out.xnorm,norm(x))
                out.iter = i
                out.info = 1
                return x,out
            end
            #only compute the next Hessian and eig if necessary
            Hx = H(x);
            E = eigen(Hx)
            v1 = E.vectors[:,1]
            lambda1 = E.values[1]
            #if very successful, expand TR radius
            if rho > eta2
               sigma = max(0.5 * sigma,sigma_min)
            end
        #unsuccessful, shrink TR radius and reuse other variables
        else
            sigma = 2 * sigma
        end
        push!(out.errHistory,errFcn(x))
        push!(out.gradnorm,norm(gx))
        push!(out.xnorm,norm(x))
    end
end
