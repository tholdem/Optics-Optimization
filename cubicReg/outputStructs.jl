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
