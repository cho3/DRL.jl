include(joinpath("..", "src", "DRL.jl"))
#import DRL: devec
using DRL
importall POMDPModels

# stuff to make things work
importall POMDPs
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions
dimensions(ipa::POMDPModels.InvertedPendulumActions) = 1
vec(ip::InvertedPendulum,a::Float64) = [a]
#devec(ip::InvertedPendulum, a::Vector{Real}) = a[1]
#import Base.convert
#convert(::Type{Float64}, a_vec::Vector{Float32}) = a_vec[1]

ip = InvertedPendulum()

ddpg = rl.DDPG()

rl.solve(ddpg, ip)
