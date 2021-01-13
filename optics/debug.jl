using LinearAlgebra,Polynomials,Printf,Random,ForwardDiff,PyPlot,PyCall,FFTW,Zygote,Optim,LaTeXStrings,FiniteDifferences
np=pyimport("numpy")

function make_prop_kernel( sz, z; dl=2,lmbd=0.5)
    nx = sz[1]
    ny = sz[2]
    k=2*pi/lmbd # wavenumber
    dkx=2*pi/((nx-1)*dl)
    dky=2*pi/((ny-1)*dl)
    kx=(LinRange(0,nx-1,nx).-nx/2)*dkx
    ky=(LinRange(0,ny-1,ny).-ny/2)*dky

    inflate(f, kx, ky) = [f(x,y) for x in kx, y in ky]
    f(kx,ky)=exp(1im*sqrt(k^2-kx^2-ky^2)*z)

    prop_kernel=inflate(f,kx,ky)

    prop_kernel = ifftshift(prop_kernel)

    return prop_kernel
end

function light_prop(e_in, prop_kernel,pfft,pifft)
    #if ndims(e_in) == 3
     #   prop_kernel = reshape(prop_kernel,(1,size(prop_kernel)...))
    #end
    ek_in  = pfft*ifftshift(e_in)
    ek_out = ek_in.*prop_kernel
    e_out  = fftshift(pifft.scale.*(pifft.p*ek_out)) #Zygote can't handle pifft directly
    return e_out
end

function phase_mod(e_in, theta; samp_ratio=1)
    #=
    e_in is the input field
    theta is the phase mask
    samp_ratio is the pixel size ratio between the phase mask and the e field
    =#
    if ndims(e_in) == 2
        if samp_ratio == 1
            e_out = e_in.*exp.(1im*theta)
        else
            e_out = e_in.*kron(exp.(1im*theta),ones((samp_ratio,samp_ratio)))
        end
    elseif ndims(e_in) == 3
        if samp_ratio == 1
            M = exp.(1im*theta)
            #e_out = e_in.*reshape(M,(1,size(M)...))
            e_out = e_in.*M
        else
            M = kron(exp.(1im*theta),ones((samp_ratio,samp_ratio)))
            #e_out = e_in.*reshape(M,(1,size(M)...))
            e_out = e_in.*M
        end
    end
    return e_out
end

#forward propagation
function propagatedOutput(e_in,airKernels,theta,pfft,pifft)
    e_out = e_in
    for jj = 1:size(theta,3)  # loop over plate
        e_out = light_prop(e_out, airKernels[:,:,jj],pfft,pifft)
        e_out = phase_mod(e_out, theta[:,:,jj], samp_ratio=1)
    end
    return e_out
end

function L(input,airKernels,e_in_post,theta,j,pfft,pifft)
    #e_in_post is the e_in after the first air propagation. It is constant so we can precompute and save time.
    output = e_in_post .* reshape(input,size(e_in_post)[1:2])
    for ii = 1:size(theta,3)  # loop over plate
        if ii == j
            output = light_prop(output, airKernels[:,:,ii+1],pfft,pifft)
        else
            output = phase_mod(output, theta[:,:,ii], samp_ratio=1)
            output = light_prop(output, airKernels[:,:,ii+1],pfft,pifft)
        end
    end
    return output
end
function L_star(input,airKernels,e_in_post,theta,j,pfft,pifft)
    #e_in_post is the e_in after the first air propagation. It is constant so we can precompute and save time.
    output = input
    for ii = size(theta,3):-1:1  # loop over plate
        if ii == j
            output = light_prop(output, airKernels[:,:,ii+1],pfft,pifft)
        else
            output = light_prop(output, airKernels[:,:,ii+1],pfft,pifft)
            output = phase_mod(output, theta[:,:,ii], samp_ratio=1)
        end
    end
    return vec(sum(e_in_post.*output,dims=3))
end

#reducing pixels resulted in very strange looking desired output
nx = 2
ny = 2
npl = 5


# wavelength
lmbd = 0.5 # in um
# pixel size
dl = 5 # unit um

x=(LinRange(0,nx-1,nx).-nx/2)*dl
y=(LinRange(0,ny-1,ny).-ny/2)*dl

#no idea what this is doing, is s_input the coordinate to create Gaussian spots?
s_input = Matrix(undef,sum(1:npl-1),2)
p = 25 # pitch
k=0
for  ii =1:npl-1
    ys = -ii*p*sqrt(2)/2
    for jj =0:ii-1
        k+=1
        xs = ys + jj*p*sqrt(2)
        s_input[k,:]=[xs,ys]
    end
end

nmod = size(s_input,1)
e_in = zeros(ny,nx,nmod)#why ny nx?
w_in = 6# beam width
inflate(f, x, y, i) = [f(xx,yy,i) for xx in x, yy in y]
#Gaussian
f(x,y,i)=exp(-((x-s_input[i,1])^2+(y-s_input[i,2])^2)/w_in^2)
for ii = 1:nmod
    e = inflate(f,x,y,ii)
    I = sum(abs2,e)
    e_in[:,:,ii] = e'/sqrt(I) #normalize, x,y very confusing!!!
end

#precompute the plan to avoid AD the plan function
pfft = plan_fft(e_in)
pifft = plan_ifft(e_in)

# output
e_target = zeros(ny, nx, nmod) #why ny nx?
w_target = 50
inflate(f, x, y) = [f(xx,yy) for xx in x, yy in y]
f(x,y)=exp(-(x^2+y^2)/w_target^2)
M = inflate(f,x,y)

for  ii = 1:npl-1
    for jj = 1:ii
        c = zeros(jj,ii-jj+1)
        c[jj,ii-jj+1]=1
        #only get the last term?
        e = np.polynomial.hermite.hermgrid2d(sqrt(2)*x/w_target, sqrt(2)*y/w_target, c).*M
        I = sum(abs2,e)
        kk = fld(ii*(ii-1),2)+jj# list index (it's actually just 1:10)
        e_target[:,:,kk] = e/sqrt(I)
    end
end

d = [2e4,2.5e4,2.5e4,2.5e4,2.5e4,2e4]
#air propagtion is a constant linear operator so we can precompute it
airKernels   = zeros(ComplexF64,(nx,ny,npl+1))
for jj = 1:npl+1
  airKernels[:,:,jj] = make_prop_kernel( (nx,ny), d[jj],dl=dl,lmbd=lmbd)
end

mat(x) = reshape(x,(nx,ny,npl))
f(x)=1/2*norm(propagatedOutput(e_in,airKernels,mat(x),pfft,pifft)-e_target)^2

function grad_f(x,mat,airKernels,e_in_post,e_target,pfft,pifft)
    g(x)=exp.(x.*im)
    X = mat(x)
    grad = zeros(size(X)[1]*size(X)[2],size(X)[3])
    for j = 1:size(X,3)
        xj = X[:,:,j]
        g_xj = g(xj)
        input = L(g_xj.*xj,airKernels,e_in_post,X,j,pfft,pifft)-(e_target.+0im)
        grad[:,j] = real(L_star(input,airKernels,e_in_post,X,j,pfft,pifft))
    end
    return vec(grad)
end

x0=randn(npl*nx*ny)
f(x0)
e_in_post = light_prop(e_in, airKernels[:,:,1],pfft,pifft)

grad_f(x0,mat,airKernels,e_in_post,e_target,pfft,pifft)
