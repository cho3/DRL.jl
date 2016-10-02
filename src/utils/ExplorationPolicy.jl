# ExplorationPolicy.jl
# mmisc stuff. May or may not put more in


abstract ExplorationPolicy <: Policy
initialize!(::ExplorationPolicy, mdp::MDP) = begin end # do nothing

type EpsilonGreedy <: ExplorationPolicy
    eps::Float64
end
EpsilonGreedy(;eps::Float64=0.85) = EpsilonGreedy(eps)

type NormalExplorer <: ExplorationPolicy
    sigma::Union{Float64, Vector{Float64}, Matrix{Float64}}
    sigma_clean::Union{Void,Vector{Float64}, Matrix{Float64}}
end
NormalExplorer(;sigma::Union{Float64, Vector{Float64}, Matrix{Float64}}
=1.) = NormalExplorer(sigma, nothing)

function next!(ne::NormalExplorer, rng::AbstractRNG)
    r = randn(rng, size(ne.sigma_clean,1))
    if isa(ne.sigma_clean,Vector)
        return ne.sigma_clean.*r
    elseif isa(ne.sigma_clean,Matrix)
        return ne.sigma_clean*r
    end
    error("Should not happen, how")
end

function initialize!(ne::NormalExplorer, mdp::MDP)
    if ne.sigma_clean != nothing
        return
    end

    a_dims = dimensions( actions(mdp) )

    if isa(ne.sigma, Matrix)
        assert( size(ne.sigma, 1) == a_dims )
        # below will break if not square
        ne.sigma_clean = cholfact(ne.sigma)
        # can do try-catch, but whatever, no point
    elseif isa(ne.sigma, Vector)
        assert( length(ne.sigma) == a_dims)
        ne.sigma_clean = ne.sigma
    elseif isa(ne.sigma, Real)
        ne.sigma_clean = ne.sigma * ones(a_dims)
    else 
        error("This should have never happened, sigma is type: $(typeof(ne.sigma))")
    end
end

# See: http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
@enum SDEStep EulerMaruyama Analytical
type OrnsteinUhlenbeckExplorer <: ExplorationPolicy
    theta::Float64
    sigma::Float64
    mu::Vector{Real}
    dt::Float64
    state::Vector{Real}
    method::SDEStep
end
OrnsteinUhlenbeckExplorer(;theta::Float64=0.15, sigma::Float64=0.2) = OrnsteinUhlenbeckExplorer(theta, sigma, [0.], 1., [0.], EulerMaruyama)
# TODO ^^ fix this up

function initialize!(oue::OrnsteinUhlenbeckExplorer, mdp::MDP)
    dim_a = dimensions(actions(mdp))
    oue.state = zeros(dim_a)
    oue.mu = zeros(dim_a)
    return oue
end

function next!(oue::OrnsteinUhlenbeckExplorer, rng::AbstractRNG)
    if oue.method == EulerMaruyama
        return __em_next!(oue, rng)
    else # oue.method == Analytical
        return __a_next!(oue, rng)
    end
end

function __em_next!(oue::OrnsteinUhlenbeckExplorer, rng::AbstractRNG)
    
    oue.state += oue.theta*(oue.mu - oue.state)*oue.dt + oue.sigma*randn(rng, size(oue.state))

    return oue.state
end

function __a_next!(oue::OrnsteinUhlenbeckExplorer, rng::AbstractRNG)
    # TODO
    error("Unimplemented")
end

