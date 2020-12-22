using Zygote, Random,ForwardDiff

f(x,y)  = 1/2*(x^2 - y^2)
sigma2  = .2;
g(x,y)  = exp(-(x^2+y^2)/sigma2);
h(x,y)  = f(x,y)*g(x,y);
h_x(x,y)  = x*g(x,y) - 1/sigma2*(x^2-y^2)*x*g(x,y);
h_y(x,y)  = -y*g(x,y) - 1/sigma2*(x^2-y^2)*y*g(x,y);


h_xx(x,y) = 2*x^4 + sigma2*(y^2+sigma2)-x^2*(2*y^2+5*sigma2);
h_xy(x,y) = 2*x*y*(x^2-y^2);
h_yx(x,y) = h_xy(x,y);
h_yy(x,y) = -2*y^4+sigma2*y^2*sigma2-sigma2^2+x^2*( 2*y^2-sigma2);
hess(x,y) = g(x,y)/(sigma2^2)*[ h_xx(x,y) h_xy(x,y); h_yx(x,y) h_yy(x,y)];

F(xvec) =  h(xvec[1],xvec[2]);
Grad(xvec) = [h_x( xvec[1], xvec[2] ); h_y( xvec[1], xvec[2] ) ];
Hess(xvec) = hess( xvec[1], xvec[2] );

x0=rand(Float64,(2,1));
fun = F(x0);
gra = Grad(x0);
hes = Hess(x0);

gra2 = gradient(F,x0);
hes2 = Zygote.hessian(F,x0);

hes3 = ForwardDiff.hessian(F,x0)
