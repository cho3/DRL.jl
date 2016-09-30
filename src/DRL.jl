# DRL.jl
# auth: Christopher Ho
# affil: None :|
# date: 8/19/2016
# desc: deep reinforcement learning for julia

module DRL

export rl
module rl # hack that MxNet uses because I'm lazy
# import ...
using POMDPs
#=
    actions
    action
    Policy
    iterator
    dimensions
    rand
    vec
=#
import GenerativeModels: initial_state, generate_sr
import Base: size, push!, peek
using MXNet

# export...


# general definitions

# include...
include(joinpath("utils","NN.jl"))
include(joinpath("utils","ExplorationPolicy.jl"))
include(joinpath("utils","ExperienceReplay.jl"))
include(joinpath("solvers","DQN.jl"))
#include(joinpath("solvers","QEC.jl"))


end # rl

end # module
