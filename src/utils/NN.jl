# NN.jl
# Misc functions for MxNet

type NeuralNetwork
    arch::mx.SymbolicNode
    ctx::mx.Context
    updater::Function # derived from  mx.AbstractOptimizer
    init::Union{mx.AbstractInitializer, Vector{mx.AbstractInitializer}}
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
                        init::Union{mx.AbstractInitializer,Vector{mx.AbstractInitializer}}=mx.XavierInitializer(),
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

is_grad_param(s::Symbol) = string(s)[end-5:end] == "weight" || string(s)[end-3:end] == "bias"

function initialize!(nn::NeuralNetwork, mdp::MDP; copy::Bool=false)
    # TODO figure out how to handle input_name
    # set up updater function (so states can be maintained)

    # turn symbols into actual computational graph with resources via c backend
    nn.exec = mx.simple_bind(nn.arch, nn.ctx, grad_req=mx.GRAD_ADD; nn.input_name=>( length( vec(mdp, create_state(mdp) ) ), 1 ) )
    
    
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

    if copy
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


function build_partial_mlp()
    # TODO there's an issue wit hthis
    arch = @mx.chain mx.Variable(:data) =>
                   mx.MLP([128, 64])
    return NeuralNetwork(arch, valid=false)
end
