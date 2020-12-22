using LinearAlgebra,Polynomials,Printf,Random,ForwardDiff,Zygote
include("cubicReg.jl")
include("ARCSubproblem.jl")


f(x,y)  = 1/2*(x^2 - y^2)
sigma2  = .2
g(x,y)  = exp(-(x^2+y^2)/sigma2)
h(x,y)  = f(x,y)*g(x,y)
h_x(x,y)  = x*g(x,y) - 1/sigma2*(x^2-y^2)*x*g(x,y)
h_y(x,y)  = -y*g(x,y) - 1/sigma2*(x^2-y^2)*y*g(x,y)


h_xx(x,y) = 2*x^4 + sigma2*(y^2+sigma2)-x^2*(2*y^2+5*sigma2)
h_xy(x,y) = 2*x*y*(x^2-y^2)
h_yx(x,y) = h_xy(x,y)
h_yy(x,y) = -2*y^4+5*sigma2*y^2-sigma2^2+x^2*( 2*y^2-sigma2)
hess(x,y) = g(x,y)/(sigma2^2)*[ h_xx(x,y) h_xy(x,y); h_yx(x,y) h_yy(x,y)]

F(xvec) =  h(xvec[1],xvec[2])
Grad(xvec) = [h_x( xvec[1], xvec[2] ); h_y( xvec[1], xvec[2] ) ]
Hess(xvec) = hess( xvec[1], xvec[2] )

errFcn(x) = min(norm(x - [0;-.447213595499958]),norm(x - [0;.447213595499958]));

x0=rand(Float64,(2,1))
x,out=cubicReg(F,x0,grad=Grad,H=Hess,errFcn=errFcn)
x2,out2=cubicReg(F,x0,errFcn=errFcn)

#SVD (nonconvex, unbounded Hessian)
#m = rand([0:1:20]);
#n = randn(Int,20);
m=30
n=20
r = min(m,n)
A = randn(Float64,(m,r))*randn(Float64,(r,n))
P,~,Q = svd(A)
#projection matrices to column spaces of true P and Q
P = P/(P'*P)*P'
Q = Q/(Q'*Q)*Q'

#matricize u and v
U(x) = reshape(x[1:m*r],(m,r))
V(x) = reshape(x[m*r+1:end],(n,r))

g(x) = U(x)*V(x)'-A
#objective function
f(x) = .5*norm(g(x))^2;
grad(x) = [vec(g(x)*V(x));vec(g(x)'*U(x))];

function boxProduct(A,B)
#if dim(A)=(m1,n1), dim(B)=(m2,n2), then dim(P)=(m1*m2,n1*n2)
    m1,n1 = size(A);
    m2,n2 = size(B);
    P = zeros(m1*m2,n1*n2);
    for i=1:m1
        for j=1:m2
            for l=1:n1
                for k=1:n2
                    P[(i-1)*m2+j,(k-1)*n1+l] = A[i,l]*B[j,k];
                end
            end
        end
    end
    return P
end
#explicit Hessian
Huu(x0) = kron(V(x0)'*V(x0),Matrix(I,m,m))
Huv(x0) = kron(V(x0)',U(x0))*boxProduct(Matrix(I,n,n),Matrix(I,r,r)) +kron(Matrix(I,r,r),g(x0))
Hvu(x0) = kron(U(x0)',V(x0))*boxProduct(Matrix(I,m,m),Matrix(I,r,r))+kron(Matrix(I,r,r),g(x0)')
Hvv(x0) = kron(U(x0)'*U(x0),Matrix(I,n,n))
H(x0) = [Huu(x0) Huv(x0);Hvu(x0) Hvv(x0)]

#see difference between computed U,V and their projections to true U,V's
#column spaces
errFcn(x) = norm(U(x)-P*U(x))+norm(V(x)-Q*V(x))
x0 = randn(Float64,((m+n)*r,1))
x,out=cubicReg(f,x0,grad=grad,H=H,errFcn=errFcn,maxIts=500)
x2,out2=cubicReg(f,x0,errFcn=errFcn,maxIts=500)
