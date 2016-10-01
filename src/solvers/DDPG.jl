# DDPG.jl
# TODO ref paper: Continuous Control Using Deep Reinforcement Learning (or something)
#=
TODO figure out clean way to allow actions to be input somewhere else in the network
e.g.: 

s = mx.Variable(:state_input)
a = mx.Variable(:action_input)

conv = @mx.chain s=>mx.Convolution(...)=>mx.Convolution(...)=>mx.FullyConnected(...)
fc_in = mx.Concat(data=[conv, a], num_args=2)
out = @mx.chain fc_in=>mx.FullyConnected(...)=>mx.SoftmaxOutput(...)
=#

type DDPG <: POMDPs.Solver
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
            critic::NeuralNetwork=build_partial_mlp(Dict{MDPInput,Symbol}(MDPState=>:state_input,MDPAction=>:action_input)),
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

    info("Please make sure the last layer of your network is not an output layer (e.g. SoftmaxOutput)")


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

function POMDPs.action{S,A}(p::DDPGPolicy{S,A}, s::S, a::A=create_action(p.mdp) )

    s_vec = vec(p.mdp, s)
    copy!( p.exec.arg_dict[p.input_name], s_vec )

    mx.forward( p.exec )

    mx.copy!(p.a_vec, p.exec.outputs[1])

    return devec(mdp, p.a_vec)

end

function action{S,A}(p::ExplorationPolicy, actor::NeuralNetwork, mdp::MDP{S,A}, s, rng, a::A=create_action(mdp)) # exploration

    s_vec = vec(mdp, s)
    copy!( actor.exec.arg_dict[actor.input_name], s_vec )

    mx.forward( actor.exec )

    a_vec = mx.copy!(zeros(Float32, size(actor.exec.outputs[1])), actor.exec.outputs[1])

    noise = next!(p, rng)

    a_vec += noise

    return devec(mdp, a_vec), s_vec, a_vec
end


function ddpg_update!(actor::NeuralNetwork, critic::NeuralNetwork, actor_target::NeuralNetwork, critic_target::NeuralNetwork, mem::ReplayMemory, disc::Real, rng::AbstractRNG, input_idx::Int)

    td_avg = 0.

    for idx = 1:actor.batch_size
        # sample memory 
        s_idx, a_idx, r, sp_idx, terminalp = peek(mem, rng=rng)

        # forward on actor_target
        mx.copy!( actor_target.arg_dict[actor.input_name], state(mem, sp_idx) )
        mx.forward( actor_target )

        # forward on critic_target (w/ prev forward as input)
        mx.copy!( critic_target.arg_dict[critic.input_name[MDPState]], state(mem, sp_idx) )
        mx.copy!( critic_target.arg_dict[critic.input_name[MDPAction]], actor_target.outputs[1] )
        mx.forward( critic_target )

        qp = mx.copy!( zeros(Float32, size( critic_target.outputs[1])), critic_target.outputs[1] )[1] # should be just one variable

        if terminalp
            q_target = r
        else
            q_target = r + disc * qp
        end

        # forward on critic (for tderror and backprop)
        mx.copy!( critic.arg_dict[critic.input_name[MDPState]], state(mem, sp_idx) )
        mx.copy!( critic.arg_dict[critic.input_name[MDPAction]], action(mem, a_idx) )
        mx.forward( critic )
        
        q = mx.copy!( zeros(Float32, size(crtic.outputs[1])), critic.outputs[1])[1]

        # calculate tderror
        td = q_target - q
        td_avg += td^2

        # set td as grad_out on critic_target
        grad_out = mx.ones(1,1) * td

        # backprop on critic
        mx.backward( critic, grad_out=grad_out )

        # accumulate these gradients elsewhere
        for (i,param) in enumerate(critic,exec.grad_arrays)
            @mx.inplace crtic.grad_arrays[i] += param
        end

        # backprop on critic w/ ones as grad_out (use mx.gradient(actions?))
        mx.backward( critic.exec, grad_out=mx.ones(1,1) )

        # forward on actor (for backprop)
        mx.forward( actor.exec )

        # backprop on actor w/ prev as grad_out
        mx.backward( actor.exec, grad_out=critic.exec.grad_arrays[input_idx]) 
    end
    # perform update on network, also clear gradients
    update!(actor)
    update!(critic, grad_arrays=critic.grad_arrays)

    #= keep for now until debugging shows this works
    # clear gradients 
    clear!(get(actor.exec).grad_arrays)
    clear!(get(critic.exec).grad_arrays)
    for grad in get(actor.exec).grad_arrays
        if grad == nothing
            continue
        end
        grad[:] = 0
    end

    for grad in get(critic.exec).grad_arrays
        if grad == nothing
            continue
        end
        grad[:] = 0
    end
    =#

    # update target networks NOTE consider making a function
    for (param, param_target) in zip( get(actor.exec).arg_arrays, actor_target.arg_arrays )
        @mx.inplace param_target .*= (1.-solver.target_refresh_rate)
        @mx.inplace param_target .+= solver.target_refresh_rate * param
    end

    for (param, param_target) in zip( get(critic.exec).arg_arrays, critic_target.arg_arrays )
        @mx.inplace param_target .*= (1.-solver.target_refresh_rate)
        @mx.inplace param_target .+= solver.target_refresh_rate * param
    end

    return sqrt(td_avg/actor.batch_size)
end


function POMDPs.solve(solver::DDPG, mdp::MDP, rng::AbstractRNG)

    # setup experience replay; initialized here because of the whole solve paradigm (decouple solver, problem)
    if solver.replay_mem == nothing
        # TODO add option to choose what kind of replayer to use
        solver.replay_mem = UniformMemory(mdp; vectorized_actions=true)
    end

    # get all actions: this is for my/computational convenience
    As = actions(mdp)

    # TODO check size of output layer -- if bad, chop off end and set nn to invalid 
    # TODO check size of input layer (critic = |S| + |A|, actor = |S|)
    # TODO make sure the output layer of the critic is just a FullyConnected (or something similar)

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
            solver.critic_target = initialize!(solver.critic, mdp, copy=true, need_input_grad=true, held_out_grads=true)
        else
            initialize!(solver.critic, mdp, need_input_grads=true, held_out_grads=true)
        end
    end

    if solver.critic.batch_size != solver.actor.batch_size
        warn("Critic Batch Size does not match actor's--will use actor's")
    end

    critic_input_idx = 0
    for (idx, arg) in enumerate(mx.list_arguments(solver.critic.arch))
        if arg == solver.critic.input_name[MDPAction]
            critic_input_idx = idx
            break
        end
    end

    if critic_input_idx == 0
        error("Sorry, messed something up with finding the critic input name")
    end


    terminalp = false
    max_steps = solver.max_steps
    ctr = 1

    for ep = 1:solver.num_epochs

        initialize!(solver.exp_pol, mdp)

        s = initial_state(mdp, rng)
        # TODO prune return args
        a, s_vec, a_vec = action(solver.exp_pol, solver.actor, mdp, s, rng) # BoundsError indexed_next (tuple.jl) -- wtf TODO
        terminal = isterminal(mdp, s)

        disc = 1.0
        r_total = 0.0
        td_avg = 0.
        step = 1

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                # TODO prune return args
                ap, sp_vec, ap_vec = action(solver.exp_pol, solver.actor, mdp, sp, rng, ap) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                #_td = r + discount(mdp) * qp - q #
                # leaving here in case there is a strong desire for prioritized experience replay--requires more overhead in DDPG framework

                # terminality condition for easy access later (possibly expensive fn)
                terminalp = isterminal(mdp, sp)

                # update replay memory
                push!( solver.replay_mem, s_vec, a_vec, r, sp_vec, terminalp, rng=rng)

                if size( solver.replay_mem ) > solver.nn.batch_size
                # only update every batch_size steps? or what?
                    td = ddpg_update!( solver.actor, solver.critic, solver.actor_target, solver.critic_target, solver.replay_mem, discount(mdp), rng )
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

                s_vec = sp_vec
                a_vec = ap_vec
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
