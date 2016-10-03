# DRL.jl
# auth: Christopher Ho
# affil: None :|
# date: 8/19/2016
# desc: deep reinforcement learning for julia

# IDK how this works
#__precompile__()

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
    solver
    MDP
=#
import GenerativeModels: initial_state, generate_sr
import Base: size, push!, peek, copy!, convert
using MXNet


# general definitions
typealias RealVector Union{Vector{Real}, Vector{Int}, Vector{Float64}, Vector{Float32}}
typealias RealMatrix Union{Matrix{Real}, Matrix{Int}, Matrix{Float64}, Matrix{Float32}}

export devec
# export...
devec{S,A}(::MDP{S,A}, ::Union{RealVector,Matrix{Real}}) = error("undefined")
## TODO ULTRA TEMPORARY HACK ##
convert(::Type{Float64}, x::Vector{Float64}) = x[1]
## TODO

# include...
include(joinpath("utils","utils.jl"))
include(joinpath("utils","NN.jl"))
include(joinpath("utils","ExplorationPolicy.jl"))
include(joinpath("utils","ExperienceReplay.jl"))
include(joinpath("utils","ConjugateGradient.jl"))
include(joinpath("solvers","DQN.jl"))
include(joinpath("solvers","DDPG.jl"))
#include(joinpath("solvers","QEC.jl"))


end # rl

end # module
