# DDPG.jl
# TODO ref paper: Continuous Control Using Deep Reinforcement Learning (or something)

type DDPG <: POMDPs.solver
    actor::NeuralNetwork
    critic::NeuralNetwork
    actor_target::Union{Void,mx.Executor}
    critic_target::Union{Void,mx.Executor}
    exp_pol::ExplorationPolicy
    max_steps::Int
    num_epochs::Int
    checkpoint_interval::Int
    verbose::Bool
    stats::Dict{AbstractString,Vector{Real}}
    replay_mem::Union{Void,ReplayMemory}
    target_refresh_rate::Real # \tau in paper

    # exception stuff from History Recorder -- couldn't hurt
    capture_exception::Bool
    exception::Nullable{Exception}
    backtrace::Nullable{Any}
end


function DDPG(;
            actor::NeuralNetwork=build_partial_mlp(),
            critic::NeuralNetwork=build_partial_mlp(),
            actor_target::Union{Void,mx.Executor}=nothing,
            critic_target::Union{Void,mx.Executor}=nothing,
            exp_pol::ExplorationPolicy=NormalExplorer(),
            max_steps::Int=100,
            num_epochs::Int=100,
            checkpoint_interval::Int=5,
            verbose::Bool=true,
            stats::Dict{AbstractString,Vector{Real}}=
                    Dict{AbstractString,Vector{Real}}(
                            "r_total"=>zeros(num_epochs),
                            "td"=>zeros(num_epochs)),
            replay_mem::Union{Void,ReplayMemory}=nothing,
            capture_exception::Bool=false,
            target_refresh_rate::Real=0.001
            )

    if !( exp_pol <: NormalExplorer )
        error("Must use normal noise for actor exploration (exp_pol=NormalExplorer())")
    end


    return DDPG(
                actor,
                critic,
                actor_target,
                critic_target,
                exp_pol,
                max_steps,
                num_epochs,
                checkpoint_interval,
                verbose,
                stats,
                replay_mem,
                target_refresh_rate,
                capture_exception,
                nothing,
                nothing
            )
end


type DDPGPolicy{S,A} <: POMDPs.Policy
    exec::mx.Executor
    input_name::Symbol
    a_vec::Vector{Real}
    mdp::MDP{S,A} # for conversion/deconversion of actions
end

function POMDPs.action(p::DDPGPolicy{S,A}, s::S, a::A=create_action(p.mdp) )

    s_vec = vec(p.mdp, s)
    copy!( p.exec.arg_dict[p.input_name], s_vec )

    mx.forward( p.exec )

    mx.copy!(p.a_vec, p.exec.outputs[1])

    return convert(A, p.a_vec)

end

function action{S,A}(p::ExplorationPolicy, actor::NeuralNetwork, mdp::MDP{S,A}, s, rng, a::A=create_action(mdp)) # exploration

    s_vec = vec(mdp, s)
    copy!( actor.exec.arg_dict[actor.input_name], s_vec )

    mx.forward( actor.exec )

    a_vec = mx.copy!(zeros(Float32, size(actor.exec.outputs[1])), actor.exec.outputs[1])

    noise = next!(p, rng)

    return convert(A, a_vec + noise)
end


function ddpg_update!()

    # TODO borrowed from DQN--needs to be modified

    for idx = 1:nn.batch_size
        s_idx, a_idx, r, sp_idx, terminalp = peek(mem, rng=rng)

        # setup input data accordingly
        # TODO abstract out to kDim input
        mx.copy!( target_nn.arg_dict[nn.input_name], state(mem, sp_idx) )

        # get target
        # TODO need discount/mdp
        mx.forward( target_nn )
        qps = vec(copy!(zeros(Float32,size( target_nn.outputs[1] ) ), target_nn.outputs[1]))
        if terminalp
            qp = r
        else
            qp = r + disc * qps[a_idx]
        end

        # setup target, do forward, backward pass to get gradient
        mx.copy!( get(nn.exec).arg_dict[nn.input_name], state(mem, s_idx) )
        mx.forward( get(nn.exec), is_train=true )
        qs = copy!( zeros(Float32, size(get(nn.exec).outputs[1])), get(nn.exec).outputs[1])
        td_avg += (qp - qs[a_idx])^2

        qs[a_idx] = qp
        mx.copy!( get(nn.exec).arg_dict[nn.target_name], qs ) 
        mx.backward( get(nn.exec) )
    end

    # perform update on network
    for (idx, (param, grad)) in enumerate( zip( get(nn.exec).arg_arrays, get(nn.exec).grad_arrays ) )
        if grad == nothing
            continue
        end
        nn.updater( idx, grad, param )
    end
    
    # clear gradients    
    for grad in get(nn.exec).grad_arrays
        if grad == nothing
            continue
        end
        grad[:] = 0
    end

    # update target network
    for (param, param_target) in zip( get(nn.exec).arg_arrays, target_nn.arg_arrays )
        mx.copy!(param_target, param)
    end


    return sqrt(td_avg/nn.batch_size)

end


function POMDPs.solve(solver::DDPG, mdp::MDP, rng::AbstractRNG)

    # setup experience replay; initialized here because of the whole solve paradigm (decouple solver, problem)
    if solver.replay_mem == nothing
        # TODO add option to choose what kind of replayer to use
        solver.replay_mem = UniformMemory(mdp)
    end

    # get all actions: this is for my/computational convenience
    As = actions(mdp)

    # TODO check size of output layer -- if bad, chop off end and set nn to invalid 
    # TODO check size of input layer (critic = |S| + |A|, actor = |S|)

    # complete setup for neural ntwork if necessary
    if !solver.actor.valid
        warn("You didn't specify an actor network or your number of output units didn't match the number of actions. Either way, not recommended")
        fc = mx.FullyConnected(name=:fc_last, num_hidden=dimensions(As), data=solver.actor.arch)
        # TODO check this
        solver.nn.arch = mx.LinearRegressionOutput(name=:output, data=fc, label=mx.Variable(:target))
        solver.nn.valid = true
    end

    if !solver.critic.valid
         warn("You didn't specify a critic network or your number of output units didn't match the number of actions. Either way, not recommended")
        fc = mx.FullyConnected(name=:fc_last, num_hidden=1, data=solver.critic.arch)
        # TODO check this
        solver.nn.arch = mx.LinearRegressionOutput(name=:output, data=fc, label=mx.Variable(:target))
        solver.nn.valid = true

    end

    # setup actor network(s)
    # TODO turn this into a function
    if isnull(solver.actor.exec)
        if solver.actor_target == nothing
            solver.actor_target = initialize!(solver.actor, mdp, copy=true)
        else
            initialize!(solver.actor, mdp)
        end
    end

    # setup critic network(s)
    if isnull(solver.critic.exec)
        if solver.critic_target == nothing
            solver.critic_target = initialize!(solver.critic, mdp, copy=true)
        else
            initialize!(solver.critic, mdp)
        end
    end


    terminalp = false
    max_steps = solver.max_steps
    ctr = 1

    for ep = 1:solver.num_epochs

        initialize!(solver.exp_pol, mdp)

        s = initial_state(mdp, rng)
        # TODO prune return args
        (a, q, a_idx, s_vec,) = action(solver.exp_pol, solver.actor, mdp, s, rng) # BoundsError indexed_next (tuple.jl) -- wtf TODO
        terminal = isterminal(mdp, s)

        disc = 1.0
        r_total = 0.0
        td_avg = 0.
        step = 1

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                # TODO prune return args
                (ap, qp, ap_idx, sp_vec,) = action(solver.exp_pol, solver.actor, mdp, sp, rng, ap) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                _td = r + discount(mdp) * qp - q

                # terminality condition for easy access later (possibly expensive fn)
                terminalp = isterminal(mdp, sp)

                # update replay memory
                push!( solver.replay_mem, s_vec, a_idx, r, sp_vec, terminalp, _td, rng=rng)

                if size( solver.replay_mem ) > solver.nn.batch_size
                # only update every batch_size steps? or what?
                    td = ddpg_update!( solver.nn, get(solver.target_nn), solver.replay_mem, discount(mdp), rng )
                end

                td_avg += td

                r_total += disc*r

                disc *= discount(mdp)
                step += 1
                ctr += 1

                s = sp
                a = ap
                q = qp
                terminal = terminalp

                # possibly remove
                a_idx = ap_idx
                s_vec = sp_vec
            end
        catch ex
            if solver.capture_exception
                solver.exception = ex
                solver.backtrace = catch_backtrace()
            else
            rethrow(ex)
            end
        end

        # update metrics
        solver.stats["td"][ep] = td_avg
        solver.stats["r_total"][ep] = r_total

        # print metrics
        if mod(ep, solver.checkpoint_interval) == 0
    
            # save model
            # TODO

            # print relevant metrics
            print("Epoch ", ep, 
                "\n\tTD: ", mean(solver.stats["td"][ep-solver.checkpoint_interval+1:ep]), 
                "\n\tTotal Reward: ", mean(solver.stats["r_total"][ep-solver.checkpoint_interval+1:ep]), "\n")

        end

    end

    # return policy
    # TODO make new exec that doesn't need to train
    return DDPGPolicy(
                    solver.actor.exec,
                    solver.actor.input_name,
                    zeros(Float32, dimensions(As)),
                    mdp
                    )


end
