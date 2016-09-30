# ExplorationPolicy.jl
# mmisc stuff. May or may not put more in


abstract ExplorationPolicy <: Policy

type EpsilonGreedy <: ExplorationPolicy
    eps::Float64
end
EpsilonGreedy(;eps::Float64=0.85) = EpsilonGreedy(eps)
