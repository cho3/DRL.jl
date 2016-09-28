# DRL.jl
# auth: Christopher Ho
# affil: None :|
# date: 8/19/2016
# desc: deep reinforcement learning for julia

module DRL

# import ...
using POMDPs
using MxNet

# export...


# general definitions

# include...
include(joinpath("utils","NN.jl"))
include(joinpath("utils","ExplorationPolicy.jl"))
include(joinpath("utils","ExperienceReplay.jl"))
include(joinpath("solvers","DQN.jl"))
include(joinpath("solvers","QEC.jl"))




end # module
