# NN.jl
# Misc functions for MxNet

# this is all I can think of of the top of my head--feel free to expand
@enum MDPInput MDPState MDPAction
type NeuralNetwork
    arch::mx.SymbolicNode
    ctx::mx.Context
    updater::Function # derived from  mx.AbstractOptimizer
    init::Union{mx.AbstractInitializer, Vector{mx.AbstractInitializer}}
    exec::Nullable{mx.Executor}
    grad_arrays::Union{Vector{mx.NDArray},Void}
    batch_size::Int # vv Fold into training options?
    input_name::Union{Symbol,Dict{MDPInput,Symbol}}
    target_name::Symbol
    save_loc::AbstractString
    valid::Bool # 
end

function NeuralNetwork(
                        arch::mx.SymbolicNode;
                        ctx::mx.Context=mx.cpu(),
                        init::Union{mx.AbstractInitializer,Vector{mx.AbstractInitializer}}=mx.XavierInitializer(),
                        opt::mx.AbstractOptimizer=mx.SGD(),
                        exec::Nullable{mx.Executor}=Nullable{mx.Executor}(),
                        grad_arrays::Union{Void,Vector{mx.NDArray}}=nothing,
                        batch_size::Int=32,
                        input_name::Union{Symbol,Dict{MDPInput,Symbol}}=:data,
                        target_name::Symbol=:target,
                        save_loc::AbstractString="dqn_policy.jld",
                        valid::Bool=true
                        )

    if !isnull(exec)
        warn("exec is defined. It may be easier to let `initialize` handle it instead")
    end

    if ctx == mx.cpu()
        info("You're running the neural network on cpu--it would be faster to run on GPU (or in parallel mode, but that's not supported)")
    end

    #info("Setting up MXNet Architecture with:") #... to finish later maybe
    # TODO check if network has a LinearRegressionOutput

    # TODO check if opt has an OptimizationState
    if !isdefined(opt, :state)
        opt.state = mx.OptimizationState(batch_size)
    end

    return NeuralNetwork(arch,
                        ctx,
                        mx.get_updater(opt),
                        init,
                        exec,
                        grad_arrays,
                        batch_size,
                        input_name,
                        target_name,
                        save_loc,
                        valid
                        )
end

is_grad_param(s::Symbol) = string(s)[end-5:end] == "weight" || string(s)[end-3:end] == "bias"

function initialize!(nn::NeuralNetwork, mdp::MDP; copy::Bool=false, need_input_grad::Bool=false, held_out_grads::Bool=false)
    # TODO figure out how to handle input_name
    # set up updater function (so states can be maintained)

    # turn symbols into actual computational graph with resources via c backend
    req = held_out_grads ? mx.GRAD_WRITE : mx.GRAD_ADD

    if need_input_grad
        nn.exec = simple_bind2(nn.arch, nn.ctx, grad_req=req; nn.input_name=>( length( vec(mdp, create_state(mdp) ) ), 1 ) )
    else
        nn.exec = mx.simple_bind(nn.arch, nn.ctx, grad_req=req; nn.input_name=>( length( vec(mdp, create_state(mdp) ) ), 1 ) )
    end
    
    # initialize parameters
    if isa(nn.init, Vector)
        for (initer, arg) in zip( nn.init, mx.list_arguments(nn.arch) )
            if arg == nn.input_name || !is_grad_param(arg)
                continue
            end
            mx.init( initer, arg, get(nn.exec).arg_dict[arg] )
        end
    else # not a vector
        for arg in mx.list_arguments(nn.arch)
            if arg == nn.input_name || !is_grad_param(arg)
                continue
            end
            mx.init( nn.init, arg, get(nn.exec).arg_dict[arg] )
        end
    end

    # clone gradient args
    if need_input_grads
        nn.grad_arrays = Union{NDArray,Void}[grad != nothing ? mx.zeros(size(grad)) : nothing for grad in nn.exec.grad_arrays]
    end

    if copy
        # TODO there might be some cases where you need the input grad, but I think you only make copies when you have a target network, so no?
        copy_exec = mx.simple_bind(nn.arch, nn.ctx, grad_req=mx.GRAD_NOP; nn.input_name=>( length( vec(mdp, create_state(mdp) ) ), 1 ) )

        for arg in mx.list_arguments(nn.arch) # shared architecture
            if arg == nn.input_name
                continue
            end
            mx.copy!(copy_exec.arg_dict[arg], get(nn.exec).arg_dict[arg])
        end

        return copy_exec
    end

end


function build_partial_mlp(inputs::Union{Symbol,Dict{MDPInput,Symbol}}=:data)
    # TODO there's an issue wit hthis
    if isa(inputs, Dict)
        input_sym = mx.Variable[mx.Variable(input) for input in values(inputs)]
        input = mx.Concat(input_sym, num_args=length(input_sym) )
    else
        input = mx.Variable(inputs)
    end
    arch = @mx.chain input =>
                   mx.MLP([128, 64])
    return NeuralNetwork(arch, valid=false, input_name=inputs)
end


# convenience
function clear!(arr::Vector{mx.NDArray})
    for x in arr
        if x == nothing
            continue
        end
        x[:] = 0
    end
end


function update!(nn::NeuralNetwork; grad_arrays::Union{Void,Vector{mx.NDArray}}=nothing)
    # TODO have a freeze_param/idx arguments?
    grads = grad_arrays == nothing ? get(nn.exec).grad_arrays : grad_arrays
    # apply update
    for (idx, (param, grad)) in enumerate(zip(get(nn.exec).arg_arrays, grads))
        if grad == nothing
            continue
        end
        nn.updater( idx, grad, param )
    end
    # clear gradients
    clear!(get(nn.exec).grad_arrays)
end
