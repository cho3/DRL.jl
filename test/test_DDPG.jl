include(joinpath("..", "src", "DRL.jl"))
using DRL
importall POMDPModels

# stuff to make things work
importall POMDPs
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions

ip = InvertedPendulum()

ddpg = rl.DDPG()

rl.solve(ddpg, ip)
