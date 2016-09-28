# ExplorationPolicy.jl
# mmisc stuff. May or may not put more in


abstract ExplorationPolicy <: Policy

type EpsilonGreedy <: ExplorationPolicy
    eps::Float64
end


