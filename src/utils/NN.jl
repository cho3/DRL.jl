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
end
# TODO constructor

function initialize(nn::NeuralNetwork)
    # TODO
end

