# NN.jl
# Misc functions for MxNet

type NeuralNetwork
    arch::mx.SymbolicNode
    ctx::mx.Context
    updater::Function # derived from  mx.AbstractOptimizer
    init::Union{mx.Initializer, Vector{mx.Initializer}}
    exec::Nullable{mx.Executor}
    batch_size::Int # vv Fold into training options?
    input_name::Symbol
    target_name::Symbol
    save_loc::AbstractString
    valid::Bool # 
end

function NeuralNetwork(
                        arch::mx.SymbolicNode;
                        ctx::mx.Context=mx.cpu(),
                        init::Union{mx.Initializer,Vector{mx.Initializer}}=mx.XavierInitializer(),
                        exec::Nullable{mx.Executor}=Nullable{mx.Executor}(),
                        batch_size::Int=32,
                        input_name::Symbol=:data,
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

    return NeuralNetwork(arch,
                        ctx,
                        updater,
                        init,
                        exec,
                        batch_size,
                        input_name,
                        target_name,
                        save_loc,
                        valid
                        )
end

function initialize(nn::NeuralNetwork, mdp::MDP)
    # TODO figure out how to handle input_name
    # set up updater function (so states can be maintained)

    # turn symbols into actual computational graph with resources via c backend
    nn.exec = mx.simple_bind(nn.arch, nn.ctx, string(nn.input_name)=size( convert( , create_state(mdp) ) ) )
    
    # initialize parameters
    if isa(nn.init, Vector)
        for (initer, name) in zip( nn.init, mx.list_arguments(nn.arch) )
            if name == nn.input_name
                continue
            end
            mx.init( initer, name, nn.exec.arg_dict[name] )
        end
    else # not a vector
        for name in nn.init, mx.list_arguments(nn.arch)
            if name == nn.input_name
                continue
            end
            mx.init( init, name, nn.exec.arg_dict[name] )
        end
    end

    return nn.exec
end


function build_partial_mlp()
    arch = @mx.chain mx.Variable(:data) =>
                   mx.MLP([128, 64])
    return NeuralNetwork(arch, valid=false)
end
