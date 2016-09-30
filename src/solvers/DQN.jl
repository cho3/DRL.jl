# DQN.jl
# built off @zsunberg's HistoryRecorder.jl
# making stuff up as I'm going along
# uses MxNet as backend because native julia etc etc

type DQN
    nn::NeuralNetwork
    target_nn::Nullable{mx.Executor}
    exp_pol::ExplorationPolicy
    max_steps::Int
    num_epochs::Int
    checkpoint_interval::Int
    verbose::Bool
    stats::Dict{AbstractString,Vector{Real}}
    replay_mem::Nullable{ReplayMemory}
    target_refresh_interval::Int

    # exception stuff from History Recorder -- couldn't hurt
    capture_exception::Bool
    exception::Nullable{Exception}
    backtrace::Nullable{Any}
end
function DQN(;
            nn::NeuralNetwork=build_partial_mlp(),
            target_nn::Nullable{mx.Executor}=Nullable{mx.Executor}(),
            exp_pol::ExplorationPolicy=EpsilonGreedy(),
            max_steps::Int=100,
            num_epochs::Int=100,
            checkpoint_interval::Int=5,
            verbose::Bool=true,
            stats::Dict{AbstractString,Vector{Real}}=
                    Dict{AbstractString,Vector{Real}}(
                            "r_total"=>zeros(num_epochs),
                            "td"=>zeros(num_epochs)),
            replay_mem::Nullable{ReplayMemory}=Nullable{ReplayMemory}(),
            capture_exception::Bool=false,
            target_refresh_interval::Int=10000
            )


    # TODO check stuff or something--leave replay memory null?
    return DQN(
                nn,
                target_nn,
                exp_pol,
                max_steps,
                num_epochs,
                checkpoint_interval,
                verbose,
                stats,
                replay_mem,
                target_refresh_interval,
                capture_exception,
                nothing,
                nothing
                )

end

type DQNPolicy{S,A} <: POMDPs.Policy
    exec::mx.Executor
    input_name::Symbol
    q_values::Vector{Float32} # julia side output - for memory efficiency
    actions::Vector{A}
    mdp::MDP{S,A}
end
# TODO constructor

function POMDPs.action{S,A}(p::DQNPolicy{S,A}, s::S, a::A=create_action(mdp) ) 
    # TODO figure out if its better to have a reference to the mdp

    # assuming that s is of the right type and stuff, means one less handle

    # move to computational graph -- potential bottleneck?
    s_vec = vec(p.mdp, s)
    mx.copy!(p.exec.arg_dict[p.input_name], convert(Array{Float32,2}, reshape(s_vec, length(s_vec), 1) ) )

    mx.forward( p.exec )

    # possible bottleneck: copy output, get maximum element
    # TODO this won't work--change shape of one of the two
    copy!( p.q_values, p.exec.outputs[1] )

    p_desc = sortperm( p.q_values, rev=true)

    # return the highest value legal action
    As = POMDPs.actions( p.mdp, s ) # TODO action space arg to keep things memory efficient
    for idx in p_desc
        a = p.actions[idx]
        if a in As
            return a
        end
    end
    
    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end



function action{S,A}(p::EpsilonGreedy, solver::DQN, mdp::MDP{S,A}, s::S, rng::AbstractRNG, As_all::Vector{A}, a::A=create_action(mdp))

    # move to computational graph -- potential bottleneck?
    s_vec = convert(Vector{Float32}, vec(mdp, s) )
    mx.copy!(get(solver.nn.exec).arg_dict[solver.nn.input_name], reshape(s_vec, length(s_vec), 1) )

    mx.forward( get(solver.nn.exec) )

    # possible bottleneck: copy output, get maximum element
    q_values = vec( mx.copy!( zeros(Float32, size(get(solver.nn.exec).outputs[1])), get(solver.nn.exec).outputs[1] ) )

    As = iterator(POMDPs.actions( mdp, s ) ) # TODO action space arg to keep things memory efficient

    r = rand(rng)
    # explore, it's here because we need all that extra spaghetti
    if r > p.eps
        rdx = rand(rng, 1:length(As))
        return (As[rdx], q_values[rdx], rdx, s_vec,)#As[rand(rng, 1:length(As))]
    end

    p_desc = sortperm( q_values, rev=true)
    q = q_values[p_desc[1]] # highest value regardless of noise or legality

    # return the highest value legal action
    for idx in p_desc
        a = As_all[idx]
        if a in As
            return (a, q, idx, s_vec,)
        end
    end
    
    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end


function dqn_update!( nn::NeuralNetwork, target_nn::mx.Executor, mem::ReplayMemory, refresh_target::Bool, disc::Float64, rng::AbstractRNG )

    # NOTE its probably more efficient to have a network setup for batch passes, and one for the individual passes (e.g. action(...)), depends on memory, I guess

    # TODO preallocate s, a, r, sp
    td_avg = 0.

    for idx = 1:nn.batch_size
        s_idx, a_idx, r, sp_idx, terminalp = peek(mem, rng=rng)

        # TODO modify to be more like nature paper (e.g. target network)
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
    if refresh_target
        for (param, param_target) in zip( get(nn.exec).arg_arrays, target_nn.arg_arrays )
            mx.copy!(param_target, param)
        end
    end


    return sqrt(td_avg/nn.batch_size)

end

function POMDPs.solve{S,A}(solver::DQN, mdp::MDP{S,A}, rng::AbstractRNG=RandomDevice())

    # setup experience replay; initialized here because of the whole solve paradigm (decouple solver, problem)
    if isnull(solver.replay_mem)
        # TODO add option to choose what kind of replayer to use
        solver.replay_mem = UniformMemory(mdp)
    end

    # get all actions: this is for my/computational convenience
    As = POMDPs.iterator(actions(mdp))

    # TODO check size of output layer -- if bad, chop off end and set nn to invalid 

    # complete setup for neural ntwork if necessary
    if !solver.nn.valid
        warn("You didn't specify a neural network or your number of output units didn't match the number of actions. Either way, not recommended")
        fc = mx.FullyConnected(name=:fc_last, num_hidden=length(As), data=solver.nn.arch)
        solver.nn.arch = mx.LinearRegressionOutput(name=:output, data=fc, label=mx.Variable(:target))
        solver.nn.valid = true
    end

    # setup policy if neccessary
    if isnull(solver.nn.exec)
        if isnull(solver.target_nn)
            solver.target_nn = initialize!(solver.nn, mdp, copy=true)
        else
            initialize!(solver.nn, mdp)
        end
    end

    terminalp = false
    max_steps = solver.max_steps
    ctr = 1

    for ep = 1:solver.num_epochs

        s = initial_state(mdp, rng)
        (a, q, a_idx, s_vec,) = action(solver.exp_pol, solver, mdp, s, rng, As) # BoundsError indexed_next (tuple.jl) -- wtf TODO
        terminal = isterminal(mdp, s)


        disc = 1.0
        r_total = 0.0


        step = 1

        td_avg = 0.

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                (ap, qp, ap_idx, sp_vec,) = action(solver.exp_pol, solver, mdp, sp, rng, As) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                _td = r + discount(mdp) * qp - q

                # terminality condition for easy access later (possibly expensive fn)
                terminalp = isterminal(mdp, sp)

                # update replay memory
                push!( get(solver.replay_mem), s_vec, a_idx, r, sp_vec, terminalp, _td, rng=rng)

                if size( get(solver.replay_mem) ) > solver.nn.batch_size
                # only update every batch_size steps? or what?
                    refresh_target = mod(ctr, solver.target_refresh_interval) == 0
                    td = dqn_update!( solver.nn, get(solver.target_nn), get(solver.replay_mem), refresh_target, discount(mdp), rng )
                end

                # TODO target network update

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
        #return r_total

    end

    # return policy
    # TODO make new exec that doesn't need to train
    return DQNPolicy(
                    get(solver.nn.exec),
                    solver.nn.input_name,
                    zeros(Float32, length(As)),
                    As,
                    mdp
                    )

end


