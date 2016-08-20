# QEC.jl
# auth: Christopher Ho
# affil: None :|
# date: 8/19/2016
# desc: POMDPs.jl implementation of Model-Free Episodic Control

#= BIG TODO BLOCK

* use kNN
* OR write some k-d tree implementation
* OR use LSH
* find/make some random projections algorithm (e.g. https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf)
* alt: PCA/CUR/SVD to find embedding
* note: need samples n >> 2^k, k = dimensionality of embedding

=#

## TODO put somewhere else, make general representation
abstract Mapper

# Base type fed into solver and used to build a policy wrapper
type QEC <: Solver
    representor::Mapper # maps state to normalized feature space learned via unsupervised learning (e.g. VAE, random projections)
    Qs::Vector{Vector{Float64}} # representation of Q in latent space TODO more efficient representation
    k::Int # for kNN
    maxQs::Int # max size of Qs
    simulator::Simulator # dictates information about the episodes to learn
    ## TODO need nb_episodes?
    problem::Union{POMDP,MDP} # will leave undefined, initialize in beginning of solve if necessary
end


# policy type for QEC: basically a pared down, frozen version of the solver
type QECPolicy <: Policy
    representation::Mapper
    Qs::Vector{Vector{Float64}}
    k::Int
    problem::Union{POMDP,MDP}
end
QECPolicy(x::QEC) = QECPolicy(x.representation, x.Qs, x.k, x.problem)


function action{S,A}( policy::Union{QECPolicy,QEC}, s::S, a::A )
    
    # return historically best action
    As = actions(policy.problem, s) # nullable _As in policy?
    Qs = estimate_value(policy, s, As)

    return As[indmax[Qs]]
end


function estimate_value{S,A}( policy::Union{QEC, QECPolicy}, s::S, As::Vector{A} )

    # map to feature space
    phi = map(policy.representation, s)
    # if explicit mapping exists, return best value
    if phi in policy.Qs
        return policy.Qs[phi]
    end

    # else use kNN approximation
    values = zeros(length(As))
    for vals in get_kNN(policy.Qs, phi, policy.k)
        values += (1./policy.k) * vals
    end

    return values

end



## TODO this is temporarily just for this particular model. in the future it will be for all episodic/batch type models
function solve( solver::QEC, problem::POMDP, policy::QECPolicy=create_policy(solver, problem) )

    # initialize whatever
    R_history = zeros(solver.simulator.max_steps)

    # train
    for ep = 1:solver.nb_episodes
        
        # initialize stuff for episode

        disc = 1.
        T = solver.simulator.max_steps
        
        # simulate an episode
        for t = 1:solver.simulator.max_steps

            a = action(problem, s)

            sp, r = generate_sr(problem, ...)

            # TODO break conditions
            S_history[t] = 0
            A_history[t] = 0
            R_history[t] = r
            
            s = sp  

            T = t # TODO put this update in break condition
        end

        
        # estimate return from each state-action pair at each timestep
        R = 0.

        for t = T:-1:1

            s = S_history[t]
            a = A_history[t]
            r = R_history[t]
            
            R = r + discount(problem) * R

            # update estimate for best Q(s,a)
            
        end

    end

    
    return QECPolicy(solver)

end

