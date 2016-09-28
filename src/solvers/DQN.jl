# DQN.jl
# built off @zsunberg's HistoryRecorder.jl
# making stuff up as I'm going along
# uses MxNet as backend because native julia etc etc


type DQN
    nn::NeuralNetwork
    exp_pol::ExplorationPolicy
    max_steps::Int
    num_epochs::Int
    checkpoint_interval::Int
    verbose::Bool
    stats::Dict{AbstractString,Vector{Real}}
    replay_mem::ReplayMemory

    # exception stuff from History Recorder -- couldn't hurt
    capture_exception::Bool
    exception::Nullable{Exception}
    backtrace::Nullable{Any}
end
# TODO constructor

type DQNPolicy{S,A} <: POMDPs.policy
    exec::mx.Executor
    input_name::Symbol
    q_values::Vector{Float32} # julia side output - for memory efficiency
    actions::Vector{A}
    mdp::MDP{S,A}
end
# TODO constructor

function action{S,A}(p::DPWPolicy{S,A}, s::S, a::A=create_action(mdp) ) 
    # TODO figure out if its better to have a reference to the mdp

    # assuming that s is of the right type and stuff, means one less handle

    # move to computational graph -- potential bottleneck?
    mx.copy!(p.exec.args_dict[p.input_name], convert(Vector{Float32}, s) )

    mx.forward( p.exec )

    # possible bottleneck: copy output, get maximum element
    copy!( p.q_values, p.exec.outputs )

    p_desc = sortperm( p.q_values, rev=true)

    # return the highest value legal action
    As = POMDPs.action( p.mdp, s ) # TODO action space arg to keep things memory efficient
    for idx in p_desc
        a = p.actions[idx]
        if a in As
            return a
        end
    end
    
    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end

function dqn_update!( nn::NeuralNetwork, mem::ReplayMemory, disc::Float64, rng::AbstractRNG )

    # NOTE its probably more efficient to have a network setup for batch passes, and one for the individual passes (e.g. action(...)), depends on memory, I guess

    # TODO preallocate s, a, r, sp
    td_avg = 0.

    for idx = 1:nn.batch_size
        s_idx, a_idx, r, sp_idx = peek(mem, rng=rng)

        # setup input data accordingly
        # TODO abstract out to kDim input
        mx.copy!( nn.exec.args_dict[nn.input_name], state(mem, sp_idx) )

        # get target
        # TODO need discount/mdp
        mx.forward( nn.exec)
        qps = copy!(zeros(size( nn.exec.outputs[1] ) ), nn.exec.outputs[1])
        qp = r + disc * qps[a_idx]

        # setup target, do forward, backward pass to get gradient
        mx.copy!( nn.exec.args_dict[nn.input_name], state(mem, s_idx) )
        mx.forward( nn.exec )
        qs = copy!( zeros(size(nn.exec.outputs[1])), nn.exec.outputs[1])
        td_avg += qp - qs[a_idx]

        qs[a_idx] = qp
        mx.copy!( nn.exec.args_dict[nn.target_name], qs ) 
        mx.backward( nn.exec )
    end
    # TODO check if consistent with nature paper
    for grad in nn.exec.grad_arrays
        if grad == nothing
            continue
        end
        @mx.inplace grad /= nn.batch_size
    end

    # perform update on network
    for (idx, (param, grad)) in enumerate( zip( nn.exec.arg_arrays, nn.exec.grad_arrays ) )
        if grad == nothing
            continue
        end
        nn.updater( idx, grad, param )
    end

    
    # clear gradients    
    for grad in nn.exec.grad_arrays
        if grad == nothing
            continue
        end
        @mx.inplace grad[:] = 0
    end


    return td_avg/nn.batch_size

end

function action(p::EpsilonGreedy, solver::DQN, mdp::MDP{S,A}, s::S, rng::AbstractRNG, a::A=create_action(mdp))

    As = POMDPs.actions( mdp, s ) # TODO action space arg to keep things memory efficient
    # explore
    if r > p.eps
        return As[rand(rng, 1:length(As))]
    end
    # otherwise, do best action

    # move to computational graph -- potential bottleneck?
    s_vec = convert(Vector{Float32}, s)
    mx.copy!(solver.nn.exec.args_dict[p.input_name], s_vec )

    mx.forward( solver.nn.exec )

    # possible bottleneck: copy output, get maximum element
    q_values = copy!( zeros(Float32, size(solver.nn.exec.outputs)), solver.nn.exec.outputs )

    p_desc = sortperm( q_values, rev=true)
    q = q_values[p_desc[1]] # highest value regardless of noise or legality


    # return the highest value legal action
    for idx in p_desc
        a = p.actions[idx]
        if a in As
            return a, q, idx, s_vec
        end
    end
    
    error("Check your actions(mdp, s) function; no legal actions available from state $s")

end

function solve{S,A}(solver::DQN, mdp::MDP{S,A}, rng::AbstractRNG=RandomDevice())

    # setup policy if neccessary
    if isnull(solver.nn.exec)
        initialize(solver.nn)
    end

    # setup experience replay
    # TODO
    
    for ep = 1:solver.num_epochs

        s = initial_state(mdp, rng)
        a, q, a_idx, s_vec = action(solver.exp_pol, solver, mdp, s, rng)

        max_steps = solver.max_steps

        disc = 1.0
        r_total = 0.0


        step = 1

        td_avg = 0.

        try
            while !isterminal(mdp, s) && step <= max_steps

                sp, r = generate_sr(mdp, s, a, rng)

                ap, qp, ap_idx, sp_vec = action(solver.exp_pol, solver, mdp, sp, rng) # convenience, maybe remove ap_idx, s_vec

                # 1-step TD error just in case you care (e.g. prioritized experience replay)
                _td = r + discount(mdp) * qp - q

                # update replay memory
                push!( solver.replay_mem, s_vec, a_idx, r, sp_vec, _td, rng=rng)

                # only update every batch_size steps? or what?
                td = dqn_update!( solver.nn, solver.replay_mem, discount(mdp), rng )

                td_avg += td

                r_total += disc*r

                disc *= discount(mdp)
                step += 1

                s = sp
                a = ap
                q = qp

                # possibly remove
                a_idx = ap_idx
                s_vec = sp_vec
            end
        catch ex
            if sim.capture_exception
                sim.exception = ex
                sim.backtrace = catch_backtrace()
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
                "\n\tTD: ", mean(solver.stats["td"][ep-solver.checkpoint_interva:ep]), 
                "\n\tTotal Reward: ", mean(solver.stats["r_total"][ep-solver.checkpoint_interva:ep]) )

        end
        #return r_total

    end

    # return policy
    # TODO

end


